# Pre-training Pythia-14M on rented home GPUs

*Gabriel Francisco dos Santos Silva. October 2026.*

Scaling a language-model run is, before anything else, a networking problem. In [synchronous data parallelism](https://arxiv.org/abs/2006.15704), every GPU trains on a slice of the batch, and the gradients are averaged before the next optimizer step. Training clusters are built around that exchange, with fast links between GPUs and between servers. Renting the same number of GPUs from people's homes gives each machine a residential internet connection to do the exchange over.

This does not get better just because the model is small. [Pythia-14M](https://arxiv.org/abs/2304.01373), the smallest model in EleutherAI's Pythia suite, has about 14 million parameters, or 56 MB in fp32. A step on a quarter of its reference batch takes about two seconds on an RTX 3090. Uploading 56 MB takes roughly six seconds at 10 MB/s and eighty seconds at 0.7 MB/s, both connection speeds encountered in this study. Either connection makes a per-step exchange longer than the computation it synchronizes.

[DiLoCo](https://arxiv.org/abs/2311.08105) lets each replica take several local optimizer steps before exchanging parameter displacements. [Streaming DiLoCo](https://arxiv.org/abs/2501.18512) spreads the exchange over fragments of the model and overlaps it with training. [Decoupled DiLoCo](https://arxiv.org/abs/2604.21428) removes the requirement that all learners reach the exchange together: a synchronizer closes a round once enough contributions have arrived, with a grace window for those a little behind.

With eight-bit deltas and compression, a replica of Pythia-14M can send about 11 MB every 25 local steps. It can keep training while that update travels, and it does not need to be reachable by another replica. If the exchange is that small, the link can be a home connection and the medium a bucket.

That is the arrangement tested here. [GradientMail](https://github.com/ppgsi-lab/gradientmail) implements Decoupled DiLoCo with object storage as the only channel between replicas. Pythia-14M starts from its published initialization and trains on the first 4,000 steps of its reference data order, divided across one to sixteen RTX 3090s rented through Salad. The experiment follows what happens to training time and held-out loss as more machines share that work, and what the rented GPU time costs.

## How the replicas communicate

GradientMail puts those exchanges in a bucket. Every machine runs a learner and a synchronizer. One synchronizer claims a round through the bucket, merges its contributions and publishes the updated fragment, called a *commit*. Renewable leases let another machine take over if the claimant stops responding. There is no separate coordinator process, and a machine behind a home router needs only outbound access to storage.

![Learners publish parameter deltas to a shared bucket. One of the training nodes merges the deltas and publishes a commit; the other nodes read it. All communication passes through the bucket.](../assets/training-flow.svg)

*Figure 1. GradientMail's communication path. The diagram illustrates the library with heterogeneous devices and RDA merging. This experiment uses RTX 3090s throughout and the weighted-mean merge described below.*

The first smoke tests ran for 64 steps on four machines, with three contributions enough to close a round. Three machines uploaded their updates in seconds. The fourth uploaded at 60 KB/s, taking seven minutes to send its first 26 MB embedding snapshot. By the time it arrived, the round had closed. That machine contributed to no round in either test.

The other three kept training, and the loss kept falling. The curve gave no indication that updates from a quarter of the batch were missing. Before attempting longer runs, admission checks were added to measure each machine's connection to the bucket, along with records of who participated in each round. That made it possible to check whether the rented machines were contributing, rather than just running.

## Experimental setup

### Reference model, data and token budget

Participation records account for the machines' work, but judging what the model learned needs a reference. Pythia publishes initial weights, intermediate checkpoints and the training recipe. Its [preshuffled dataset](https://huggingface.co/datasets/EleutherAI/pile-standard-pythia-preshuffled) also exposes the reference data order. Starting from those weights and sequences makes it possible to compare the distributed result with a published checkpoint after the same nominal amount of training.

Every run starts from `EleutherAI/pythia-14m` at step 0 and targets the first 4,000 dataset steps. A dataset step contains 1,024 sequences of 2,048 predicted tokens, giving a nominal budget of 8,388,608,000 tokens. This is an early segment of Pythia's 143,000-step training run, not a reproduction of the complete run.

With $n$ replicas, each local batch contains $1024/n$ sequences. A *dataset step* therefore identifies the reference data and learning rate; it does not mean that all replicas update one shared model synchronously. Each replica applies AdamW to its own slice before the parameter updates are merged.

Work is assigned through a bucket-backed queue in units of twenty consecutive dataset steps for one slice. Faster machines claim more units, which reduces the wait for the slowest machine at the end. The queue preserves the reference sequence indices and learning rate, but asynchronous execution does not preserve one global update order. A unit claimed by a machine that dies is not replayed, so the nominal token budget is not a guarantee that every sequence contributes to the final model.

The model follows Pythia-14M's GPT-NeoX configuration: six layers, hidden width 128, four attention heads, rotary embeddings on a quarter of each head, and separate input and output embeddings over 50,304 tokens. The two embeddings contain 12.9 million of its 14.1 million parameters.

![Pythia-14M's input embedding, six GPT-NeoX blocks, final layer norm and untied output projection. The expanded block shows attention and MLP in parallel, with separate layer norms and a shared residual connection. The two vocabulary tables hold about 92% of the parameters.](../assets/pythia-architecture.svg)

*Figure 2. Pythia-14M's architecture. Each block adds the attention and MLP outputs to the same input residual. The input embedding and output projection have separate weights and together contain 12.9 million parameters.*

The inner optimizer follows the reference recipe: AdamW, weight decay 0.1 on matrices, betas 0.9 and 0.95, and gradient clipping at norm 1. The learning rate rises linearly to $10^{-3}$ over 1,430 steps, then follows the reference cosine schedule toward $10^{-4}$ at step 143,000. The runs use bf16 autocast; the reference used fp16 with loss scaling.

Before the distributed runs, the implementation's logits were checked against `transformers`' GPT-NeoX and differed by at most $7 \times 10^{-5}$. A sixteen-step replay without outer synchronization ended within 0.003 of the official step-16 loss. These checks cover the model and the beginning of the training pipeline, not equivalence over 4,000 steps.

### Hardware and synchronization

The main experiment uses the following configuration at 2, 4, 8 and 16 replicas:

1. **Hardware.** Each machine has one RTX 3090 with 24 GB, rented through [Salad](https://salad.com) and provisioned by [Skyward](https://github.com/gabfssilva/skyward). The listed price at the time of the runs was US$0.17 per GPU-hour. Admission requires a marketplace listing of at least 300 Mbit/s down and 200 up, followed by a 26 MB transfer check at 20 Mbit/s or better.

2. **Mailbox backend.** A [Hugging Face Storage Bucket](https://huggingface.co/docs/hub/storage-buckets), accessed through its S3-compatible API, carries all communication between replicas.

3. **Experimental setup.**

   1. **Fragment schedule.** $H = 25$ local steps per fragment, with four fragments and at most two rounds in flight.
   2. **Fragment split.** One fragment per embedding, with the transformer blocks split into two halves.
   3. **Quorum.** $n - 1$ contributions, or one when $n \le 2$, followed by an adaptive grace window.
   4. **Merge.** Weighted mean, with contribution weight $w_i = t_i^2/s_i$, where $t_i$ is tokens and $s_i$ is local steps.
   5. **Outer optimizer.** Nesterov momentum, learning rate 0.7 and momentum 0.9.
   6. **Alignment.** `Carry`, preserving local work performed while an update is in flight.
   7. **Exchange format.** `delta_dtype="F8_E4M3"` for learner deltas and `commit_dtype="F8_E4M3"` for global commit updates. All payloads use zstd compression; full fragment weights and outer optimizer state remain fp32.

The one-replica baseline comes from an earlier run, using `Carry`, $H = 50$ and fp32 exchange with the same inner and outer optimizers. Its interval and exchange format differ from the main configuration. Replica count also changes local batch size, so this is not a controlled measurement of replica count alone.

`Carry` handles the work done during communication. Let $s$ be the fragment a learner published, $\theta$ its current local fragment, and $\Theta'$ the new commit. Alignment sets

$$
\theta \leftarrow \Theta' + (\theta - s).
$$

In words, the learner applies its progress since publication on top of the shared update. That progress enters its next contribution. This is a choice made by GradientMail, not a claim that an update computed at the old parameters equals one computed at the new parameters.

Two different updates pass through the bucket. Each learner uploads a **learner delta**, the displacement produced by its local training. The synchronizer reads those deltas, combines them and applies the outer optimizer. It then publishes a commit containing the new fragment weights and optimizer state, plus a **global commit update**: the displacement from the previous shared fragment to the new one. A learner holding the previous version can download this global commit update and reconstruct the new weights. A learner without that version downloads the full fragment weights.

`delta_dtype` controls the learner's upload; `commit_dtype` controls the optional global commit update. They are independent: FP4 learner deltas can be combined with `commit_dtype=None`, which publishes full fp32 weights and momentum without a separate global commit update. Setting `commit_dtype="F8_E4M3"` adds an FP8 global commit update and a separate fp32 master copy. The master preserves the value before quantization rounding; the published weights and momentum also remain fp32. The main runs use FP8 in both directions. Their compressed learner deltas are about 5.2 MB per embedding and 0.5 MB per block fragment, compared with 25.8 and 2.4 MB for uncompressed fp32 tensors.

### Evaluation protocol

The reported metric is **mean cross-entropy in nats per token**, with lower values indicating lower prediction loss. Every model is evaluated on the same 1,024 sequences from dataset step 140,000, about 2.1 million predicted tokens. This batch is outside the first 4,000 steps of the reference order. Evaluation runs on an RTX 3090 after training, using the newest commit of each fragment.

The official step-4,000 checkpoint and nine additional [PolyPythia](https://arxiv.org/abs/2503.09543) checkpoints are evaluated on that batch for comparison. The additional checkpoints provide context for variation among published reference runs, but they do not estimate uncertainty for GradientMail. Their data orders differ, and the experiment script notes that about 3% of this evaluation batch can occur in their first 4,000 training steps.

The one-replica baseline also needs a distinction: it uses GradientMail's outer Nesterov optimizer. It is not plain AdamW training. Comparing distributed runs with this baseline asks how the distributed configuration changes the result; comparing them with the official checkpoint also changes the optimizer and numerical precision. Lower loss than the checkpoint cannot be attributed to distribution alone.

**Training time** runs from the first training step to the last commit. Speed-up divides the one-replica time by that duration; parallel efficiency divides speed-up by replica count. **Training GPU cost** multiplies duration by replica count and the listed US$0.17 hourly rate. It excludes provisioning, failed allocation attempts, storage and transfer charges, and time that machines remain allocated outside training. It is not a complete invoice.

There is one run per configuration and one evaluation batch. No confidence interval or significance threshold is estimated. The study also does not measure downstream task performance, full-training convergence or models larger than Pythia-14M.

---

## Results

The comparison covers five runs, with 1, 2, 4, 8 and 16 replicas. Table 1 reports their final held-out loss alongside training time and GPU cost; Figure 3 follows their local training loss over the dataset steps.

### Training time

With the data and evaluation fixed, the runs divide the reference batch into progressively smaller slices. Table 1 follows that progression from one GPU to sixteen. The distributed runs use the configuration above; the single-replica baseline uses $H = 50$ and fp32.

| Replicas | Sequences per local step | Held-out loss | Training time | Speed-up | Parallel efficiency | Training GPU cost |
|---|---|---|---|---|---|---|
| 1 | 1,024 | 3.585 | 7 h 52 min | 1.0× | 100% | US$1.34 |
| 2 | 512 | 3.559 | 4 h 08 min | 1.9× | 95% | US$1.41 |
| 4 | 256 | 3.605 | 2 h 09 min | 3.7× | 91% | US$1.46 |
| 8 | 128 | 3.601 | 1 h 12 min | 6.6× | 82% | US$1.63 |
| 16 | 64 | 3.749 | 37 min | 12.8× | 80% | US$1.68 |

*Table 1. Training time, held-out loss and training GPU cost at each replica count, for the nominal first 4,000 dataset steps. Costs exclude provisioning and storage. Speed-up and efficiency are rounded from the recorded durations.*

Eight replicas finish 6.6 times faster than one, and sixteen finish 12.8 times faster. Communication overlaps with training, so the machines continue computing while updates move through storage. These results measure the complete training configuration, including its work queue and asynchronous synchronization; they do not isolate the overhead of the bucket against another transport.

### Training loss and held-out loss

Figure 3 compares learning progress over the same reference data. Each curve shows the mean local training loss in 100-step bins, with the initial drop above 5 omitted so the later differences remain visible. Two replicas have the lowest loss in the displayed range; the sixteen-replica curve remains above the others.

![Mean local training loss against dataset step for 1, 2, 4, 8 and 16 replicas, with an explicit legend and labels at the final logged bin. The single-replica baseline uses H = 50 and fp32; distributed runs use H = 25 and F8 with zstd.](../assets/pythia-loss.svg)

*Figure 3. Local training loss by dataset step. The dots mark the last logged 100-step bin, centered at step 3,950. These curves measure loss on the learners' own data slices; the held-out losses in Table 1 evaluate the final committed model on a separate batch.*

Finishing the assigned data sooner is useful if the resulting model has also learned from it. Four and eight replicas score 3.605 and 3.601 on the held-out batch, respectively, against 3.585 for one replica. At the precision reported in Table 1, those gaps are 0.020 and 0.016 nats per token. Two replicas score 3.559, a gap of -0.026. Without repeated runs, these values do not establish equivalence or a benefit from using two replicas.

For context, the official step-4,000 checkpoint scores 3.781 on the same evaluation batch. The nine PolyPythia checkpoints score between 3.732 and 3.762, with mean 3.741 and sample standard deviation 0.009. The sixteen-replica run scores 3.749, within that range and 0.164 above the single-replica baseline. The reference comparison includes the optimizer and precision differences described in the setup; the result within GradientMail is that the shortest training time comes with higher held-out loss than the four- and eight-replica runs.

The sixteen-replica result also has a scheduling limit. A configured $H = 25$ does not guarantee a contribution every 25 steps: rounds still take time to collect, merge and commit. Mean contribution lengths were 25, 33, 35 and 74 steps at 2, 4, 8 and 16 replicas. At sixteen replicas, the run therefore measures both a smaller local batch and a longer effective synchronization interval.

The synchronizer processes one round at a time. FP8 reduces contribution deltas and commit updates, but each commit also publishes the full fragment, outer momentum and master copy in fp32. All payloads are zstd-compressed before upload, so the synchronizer still transfers more state than a learner publishing a delta. At sixteen replicas a round closed about every ten seconds while each learner trained about two steps per second. This limits synchronization frequency even though the learners' own deltas are small. It is a candidate explanation for the higher loss, not a separately tested cause.

### Paying for the shorter run

Adding GPUs mainly trades elapsed training time for aggregate GPU hours. The one-replica run uses about 7.9 GPU-hours; the sixteen-replica run uses about 9.9. At the recorded hourly price, that puts the runs between US$1.34 and US$1.68, an increase of about 25% across the endpoints. The training GPU cost rises with replica count across these five runs.

Those figures begin once all machines are ready to train. Assembling the sixteen-replica run took 68 minutes before its 37 minutes of training. One slot went through six machines during connection checks, and another host could not reach its package mirrors. The first start barrier timed out with fourteen nodes ready; training was relaunched on the same machines. Provisioning took longer than training and adds rented time outside Table 1's accounting.

### Quantization and compression

Payload size is measured by re-encoding one retained output-embedding commit, round 456 of the eight-replica $H = 25$ run. Its anchor, fp32 master and momentum are held fixed while the update is encoded in FP32, BF16, FP8 or FP4, each with and without zstd. The counts include tensor headers and quantization scales but exclude the round's JSON record. This measures bytes for one sample, without repeating training or evaluating the alternative formats' losses.

{% include_relative _pythia-data-flow.html %}

*Interactive diagram. One fragment round with four learners. The delta and commit formats can be selected independently, and the sizes shown before or after zstd; the main experiment uses FP8 with zstd for both. FP32 commit selects the default layout, with no separate global commit update. Values on the connections include tensor headers and scales and exclude control messages. The four contributions and transfer order are schematic; the synchronizer is a role held by one of the four nodes.*

In the retained FP8 commit, a learner holding the previous fragment downloads a **5.29 MB global commit update** instead of **17.20 MB of full fragment weights**, both after zstd. It does not download the master or momentum. The synchronizer uploads all three payloads, including the weights and optimizer state, for a **65.98 MB complete commit** (83.71 MB before zstd).

The default fp32 commit layout omits the separate global commit update and master, reducing its compressed total to 43.49 MB in the replay. Quantizing the global commit update therefore saves bytes on each learner's download while increasing the synchronizer's total upload in this implementation. Whether that trade improves round time depends on which connections limit the exchange; the payload replay does not measure that effect.

![Payload sizes for one Pythia-14M output-embedding global commit update in FP32, BF16, FP8 and FP4, with and without zstd. The upper group shows the global commit update a learner can download; the lower group shows the synchronizer's complete commit upload. Quantized commit updates require a separate fp32 master and update payload.](../assets/pythia-payloads.svg)

*Figure 4. Re-encoding the same global commit update and complete commit. The upper group measures global commit updates, not learner deltas. Its standalone FP32 update is 19.58 MB after zstd; the default commit layout publishes full weights instead of this update. FP8 is E4M3; FP4 includes one scale per 32 elements. Compression uses byte shuffling and zstd level 1. The FP8 encoding reproduces the stored payloads exactly. These are sizes for one retained commit, not averages across the training runs.*

FP4 with zstd reduces the global commit update further, to 3.25 MB, but the complete commit still takes 63.94 MB. The fp32 state limits how much shrinking the global commit update can reduce the total in the current commit layout.

## What happened on the home machines

Passing admission also does not make a machine's performance constant for the rest of the run. The marketplace assigns the hosts, and identical GPU models can have different power limits and connections. Uploads across the study ranged from 0.7 to 27 MB/s, so the rounds still needed to accommodate machines that fell behind after training started.

During the main sixteen-replica run, one node disappeared at 83% of training. The remaining fifteen kept closing rounds under the configured quorum. In the four-replica run, a power-limited 3090 missed about 90 of 509 rounds, and its retained local work entered later contributions through `Carry`. These are observed recoveries, not a controlled fault-injection study or an estimate of failure probability.

Storage can delay progress too. In an earlier two-replica run, a synchronizer listed no snapshots for 199 seconds, including its own, before the other node took over and closed the round. Leases and takeovers let the run recover. The mailbox removes direct connections between learners, but makes storage latency and visibility part of the training system.

## What the supporting runs explain

The configuration used for the scaling experiment came from fifteen runs exploring replica count, synchronization interval, alignment and exchange precision. Its four distributed runs are part of that exploration, rather than an independent confirmation of a configuration fixed in advance. The supporting measurements in the appendix help explain why the configuration combines short intervals with compressed updates and retained local work.

At eight replicas, `Carry` scores 3.601 with $H = 25$ and eight-bit exchange, 3.678 with $H = 50$ and fp32, and 3.836 with $H = 250$ and bf16. More frequent synchronization is a plausible contributor to the lower loss, consistent with DiLoCo's interval experiments. However, precision and machine behavior also differ. These runs do not isolate the causal effect of $H$.

Preserving overlapped work has a direct accounting benefit. At eight replicas and $H = 50$, `Carry` retains 98.5% of fragment-steps, against 55% for `Overwrite`, which discards progress made after publication. Their held-out losses are 3.678 and 3.723. That supports the choice to retain work in this setting, but `Overwrite` was not tested at $H = 25$. Nor was fp32 at $H = 25$, so the main experiment does not measure the loss cost of its eight-bit exchange.

## What this experiment leaves open

The measured scaling applies to a 14-million-parameter model whose embeddings account for about 92% of parameter bytes. Each GPU holds the entire model and optimizer state. This arrangement increases throughput; it does not distribute a model that is too large for one GPU's memory. Larger models, longer runs and other fragment layouts need their own measurements. [OpenDiLoCo](https://arxiv.org/abs/2407.07852) reproduced DiLoCo across continents, and [INTELLECT-1](https://arxiv.org/abs/2412.01152) trained a 10-billion-parameter model with distributed contributors, but those results do not measure the storage-only configuration tested here.

The evaluation also has a finite endpoint. Each model is assembled from its newest fragment commits, whose final rounds can have different participant counts; the main sixteen-replica run's last round had five participants. Repeated runs and evaluation on additional unseen data would be needed to estimate variability and test whether the observed ordering persists.

The gain over the official checkpoint remains unattributed. The one-replica run already scores 3.585, below the checkpoint's 3.781, so multiple replicas are not necessary for that gap. The [original DiLoCo study](https://arxiv.org/html/2311.08105v3#S6.SS1) also reports improvements with a single replica in its own setting. Here, a full one-replica replay with an identity outer step would help separate the outer optimizer from numerical precision. A one-replica run at $H = 25$ would also make the scaling baseline match the main interval.

## Closing

The bucket gives these machines a place to exchange updates without making them reachable by one another. As more GPUs join, the rate at which those updates become commits starts to matter: the sixteen-replica run took about three times the configured interval between contributions. Testing faster commits at that scale would help separate this scheduling limit from the effect of dividing training across more replicas.

## Reproduction and sources

The model, data reader, provisioning configuration, training loop and evaluator are in [`examples/pythia.py`](../../examples/pythia.py). The [scaling figure source](../figures/pythia-scaling.py) contains the recorded loss and timing summaries used by the plots. These are transcribed summaries, not the raw console logs or round records. The separate payload-size comparison has a [measurement script](../figures/pythia-payloads.py) and [byte counts with source hashes](../figures/pythia-payloads.json).

The example documents bucket credentials and provider setup. After that setup, the eight-replica configuration can be launched with:

```sh
PYTHIA_NODES=8 sky run examples/pythia.py train --node all \
    --steps 4000 --prefix pythia-14m-8n-h25-f8 \
    --quorum 7 --sync-every 25 --alignment carry \
    --delta-dtype F8_E4M3 --commit-dtype F8_E4M3 --compression zstd
```

Evaluation uses the same compute configuration and bucket prefix:

```sh
PYTHIA_NODES=8 sky run examples/pythia.py compare --node 0 \
    --step 4000 --prefix pythia-14m-8n-h25-f8
```

The commands specify the recipe, not a deterministic replay of marketplace assignments or asynchronous events. The example keeps compute allocated between commands; its setup notes describe deletion when finished. This post does not record exact dependency versions or a code revision for each historical run, which limits archival reproducibility.

## Appendix: supporting measurements

### Synchronization and participation in the main experiment

"Steps kept" is the fraction of trained fragment-steps retained for committed contributions, not a count of distinct dataset tokens. A round concerns one fragment; the parenthesized count is rounds with every live node participating. Upload figures are per-node learner delta traffic and exclude the synchronizer's commit traffic.

| Replicas | Steps kept | Rounds (all live nodes) | Mean steps per contribution | Delta upload per node |
|---|---|---|---|---|
| 1, $H = 50$, fp32 | 99.5% | 320 (320) | 50 | Not reported |
| 2 | 99.7% | 630 (625) | 25 | 1.80 GB |
| 4 | 99.5% | 509 (415) | 33 | 1.45 GB |
| 8 | 99.5% | 458 (436) | 35 | 1.31 GB |
| 16 | 99.0% | 218 (150) | 74 | 0.62 GB |

*Table A1. Synchronization records for the runs in Table 1. A smaller upload total can reflect fewer rounds and a longer effective interval, rather than faster communication.*

At four replicas, the node closing the first 63 rounds had a 23 Mbit/s uplink and took 21 seconds per embedding commit. Contributions averaged about 46 steps in that period and 29 after a faster node took over. At eight replicas, a round closed every 7.5 seconds and contributions averaged 35 steps.

### Eight-replica comparisons

| Alignment | Configured $H$ | Exchange | Held-out loss | Steps kept | Mean steps per contribution | Delta upload per node | Training time |
|---|---|---|---|---|---|---|---|
| `Carry` | 25 | `F8_E4M3` + zstd | 3.601 | 99.5% | 35 | 1.31 GB | 1 h 12 min |
| `Carry` | 50 | fp32 | 3.678 | 98.5% | 63 | 3.66 GB | 1 h 09 min |
| `Adaptive` | 50 | fp32 | 3.705 | 70% | 41 | Not reported | 1 h 10 min |
| `Overwrite` | 50 | fp32 | 3.723 | 55% | 34 | Not reported | 1 h 08 min |
| `Carry` | 50 | `F8_E4M3` + zstd | 3.733 | 99.2% | 78 | 0.62 GB | 1 h 10 min |
| `Blend(0.5)` | 50 | fp32 | 3.770 | 75%† | 56 | Not reported | 1 h 14 min |
| `Carry` | 250 | bf16 | 3.836 | 96.8% | 227 | 0.48 GB | 1 h 10 min |

*Table A2. Supporting runs on the same nominal dataset budget and local batch size. † `Blend(0.5)` retains half the overlapped displacement; its accounting counts half the overlapped steps as kept.*

`Overwrite` replaces the local fragment with the commit. `Blend(0.5)` averages the local fragment and the commit. `Adaptive` uses `Carry` when the round's mean delta cosine is at least 0.6, switches to `Overwrite` below 0.4, and retains its previous mode between those thresholds.

![Held-out loss for the seven eight-replica configurations in Table A2, with the single-replica baseline, official checkpoint and PolyPythia range for context.](../assets/pythia-ablation.svg)

*Figure A1. The eight-replica comparisons. Different intervals, exchange formats and execution conditions prevent treating every pair as a controlled ablation.*

The $H = 250$ run included a node power-limited to 150 W, which trained 1,740 steps against 3,700 to 4,720 on the others, and ended with a three-participant round. The $H = 50$ eight-bit run had repeated connectivity failures during its first fifth, followed by contributions of 200 steps or more. Its 0.055 loss gap against fp32 therefore combines exchange precision with different execution conditions. No fp32 run at $H = 25$ isolates precision within the main configuration.

At $H = 50$, median overlapped work was 28 steps at eight replicas, against mean contribution length 63. At $H = 25$, those figures were 16 and 35. They describe the scale of retained work, but a median divided by a mean is not an exact aggregate fraction of training.

At four replicas and $H = 50$, `Overwrite` scored 3.634 against 3.689 for `Carry`, reversing the eight-replica ordering. The former used an earlier library version and closed 337 of 348 rounds with all four nodes; the latter included its slowest node alongside all others in only 61 of 282 rounds. That comparison does not establish which alignment rule would produce lower loss under matched conditions.

### Earlier scaling runs

| Replicas | Main configuration loss | $H = 50$, fp32 loss | $H = 50$, fp32 training time |
|---|---|---|---|
| 1 | Not run | 3.585 | 7 h 52 min |
| 2 | 3.559 | 3.584 | 4 h 35 min |
| 4 | 3.605 | 3.689 | 2 h 15 min |
| 8 | 3.601 | 3.678 | 1 h 09 min |
| 16 | 3.749 | 3.847 | 36 min |

*Table A3. `Carry` scaling runs at the two configurations. The main configuration changes both interval and exchange precision.*

![Held-out loss versus replica count for the main configuration and the earlier H = 50 fp32 series, with the official checkpoint and PolyPythia range.](../assets/pythia-scaling.svg)

*Figure A2. Held-out loss by replica count. The reference band describes the published seeds, not a confidence interval for these runs.*

![Mean pairwise cosine similarity between contribution deltas over the H = 50 runs. Similarity declines during training and differs by replica count and alignment rule.](../assets/pythia-cosine.svg)

*Figure A3. Mean delta cosine, averaged over each tenth of the $H = 50$ runs. `Blend(0.5)` falls to about 0.06 after the sixth tenth. Cosines across different synchronization intervals are not directly comparable because contribution lengths change.*

## References

1. Biderman, S., et al. [Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling](https://arxiv.org/abs/2304.01373). *ICML*, 2023.
2. Douillard, A., et al. [DiLoCo: Distributed Low-Communication Training of Language Models](https://arxiv.org/abs/2311.08105). 2023.
3. Douillard, A., et al. [Streaming DiLoCo with overlapping communication: Towards a Distributed Free Lunch](https://arxiv.org/abs/2501.18512). 2025.
4. Douillard, A., et al. [Decoupled DiLoCo for Resilient Distributed Pre-training](https://arxiv.org/abs/2604.21428). 2026.
5. Jaghouar, S., et al. [OpenDiLoCo: An Open-Source Framework for Globally Distributed Low-Communication Training](https://arxiv.org/abs/2407.07852). 2024.
6. Jaghouar, S., et al. [INTELLECT-1 Technical Report](https://arxiv.org/abs/2412.01152). 2024.
7. van der Wal, O., et al. [PolyPythias: Stability and Outliers across Fifty Language Model Pre-Training Runs](https://arxiv.org/abs/2503.09543). *ICLR*, 2025.
8. EleutherAI. [Pythia training code and configuration](https://github.com/EleutherAI/pythia); [Pythia-14M checkpoints](https://huggingface.co/EleutherAI/pythia-14m); [preshuffled training data](https://huggingface.co/datasets/EleutherAI/pile-standard-pythia-preshuffled).
9. Hugging Face. [Storage Buckets documentation](https://huggingface.co/docs/hub/storage-buckets).
10. GradientMail. [Experiment implementation](../../examples/pythia.py) and [figure data summaries](../figures/pythia-scaling.py).
11. Li, S., et al. [PyTorch Distributed: Experiences on Accelerating Data Parallel Training](https://arxiv.org/abs/2006.15704). *VLDB*, 2020.
