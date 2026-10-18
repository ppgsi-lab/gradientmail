# Pythia-14M on four rented GPUs and a bucket

*October 2026.*

Training on rented consumer GPUs means working with machines that do not necessarily advance at the same pace. Even when every machine has an RTX 3090, its host resources and internet connection can differ. A node may process batches quickly but take too long to upload its updates; another may communicate well but train more slowly than the rest.

Imagine dividing a dataset equally between four such nodes. Waiting for all four to finish each round makes the faster machines depend on the slowest. Allowing a round to proceed with three avoids that wait, but a fourth node whose updates always arrive late may contribute nothing. A falling loss curve can look reasonable in either case.

This experiment uses four RTX 3090s rented from Salad to test GradientMail under those conditions. The nodes communicate only through a Hugging Face Storage Bucket. Each trains its own model and writes updates to the bucket; the training nodes also coordinate the merging, so there is no separate coordinator process.

To assess the result, the run starts from a published model's initial weights and processes the same data as its reference training run. That provides two things to check: whether all four nodes contribute, and how the resulting model compares with the published checkpoints.

## Choosing a reference run

EleutherAI's Pythia suite makes that comparison possible. It publishes the training recipe and intermediate weights, along with the dataset order. Starting from `pythia-14m` at step 0 therefore gives the experiment a known initialization and checkpoints to compare against as training progresses.

Pythia-14M has six layers of width 128 and 14 million parameters, including 12.9 million in its two embeddings. The official run trains for 143,000 steps, each covering 1,024 sequences of 2,048 tokens. This experiment covers the first 4,000 dataset steps, about 8.4 billion tokens.

The data comes from the preshuffled Pile. Dataset step `s` contains sequences `1024 * s` through `1024 * s + 1023`, so the four nodes can divide each global batch into slices of 256 sequences without changing which data the run covers. Every slice uses the official learning rate for its dataset step.

The reference checkpoints are available at step 0, at powers of two from 1 through 512, and every 1,000 steps afterwards. Nine additional PolyPythia runs of the smallest models use different initializations and data orders, providing a measure of variation between reference runs.

Before involving multiple machines, the model and data pipeline were checked locally. The implementation's logits match `transformers`' GPT-NeoX within 7e-5, and a 16-step replay finishes within 0.003 of the official step-16 loss. These checks help distinguish a model or data-reader error from a difference introduced by distributed training.

## Getting updates through the bucket

Skyward provisions the four Salad nodes at a listed price of about US$0.17 per GPU-hour. They are consumer machines on residential connections, assigned by the marketplace. The compute configuration and GPT-NeoX implementation live in [`examples/pythia.py`](../../examples/pythia.py), together with the `train`, `evaluate` and `compare` commands. The data reader fetches the required sequences from the Hub through HTTP range requests.

Each node runs local optimizer updates and periodically publishes a parameter snapshot to the bucket. One of the training nodes collects snapshots and records the merged result for a round. Other nodes can take over that role. Claims record who owns the round, and renewable leases let nodes track whether an owner is still present; both are objects in the same bucket.

The configuration is `gm.DiLoCo(sync_every=50, fragments=4, quorum_size=3)`. Parameters are split into four fragments, with each node publishing one fragment every 12 or 13 local steps. Each embedding occupies a 25.8 MB fragment in fp32, and the transformer blocks occupy the other two. Three participants are enough to close a round, allowing training to continue when a fourth node falls behind.

![Pythia-14M in fp32 is 56.3 MB, split into four fragments: each embedding is a 25.8 MB fragment and the six transformer blocks share the other two, about 2.4 MB each. Rounds rotate through the fragments every 12 or 13 steps with sync_every = 50, so a node uploads 25.8 MB, 25.8 MB, 2.4 MB and 2.4 MB in turn; keeping up takes about 1 MB/s.](../assets/pythia-fragments.svg)

Two rounds can be in flight while nodes continue training. The merge uses a token-weighted mean of the deltas and an outer Nesterov optimizer with learning rate 0.7 and momentum 0.9. This changes the optimization path from the reference, which averages gradients after every global batch: each distributed node applies updates to its own 256-sequence slice before merging parameters.

The first two smoke tests ran for 64 steps. Their losses fell, but one node uploaded at only 60 KB/s. Its first 26 MB snapshot took seven minutes to arrive. By then, its round had closed and the corresponding objects had been cleaned up. That node missed every round in both tests.

The other three nodes had enough participants to keep closing rounds. They learned from their own data while updates from a quarter of every batch never reached the shared model. The loss curve did not reveal the missing contribution, and the library had no milestone logs to show it directly.

![A 64-step smoke test on four nodes with quorum_size = 3. Nodes 0, 1 and 2 publish their snapshots in seconds, and every round closes with the three of them. Node 3 uploads at 60 KB/s: its first 26 MB snapshot takes about seven minutes, and lands after its round has closed and the round's objects have been cleaned up. Node 3 is in no round.](../assets/pythia-slow-upload.svg)

Stopping the run exposed another problem. The node closing rounds exited while still holding a claim on the next one. A follower waited 150 seconds to take over, then encountered the slow node's unreleased lease. Because it had not tracked that lease before taking over, it trusted it for another 120 seconds. The combined wait reached the learner's 270-second timeout for its final round. The learner gave up seconds before the round would have committed.

![Timeline of the shutdown in the smoke test. At time zero the node closing rounds exits while it still holds the claim on the next round. A follower waits 150 seconds without a record, takes over, then meets the slow node's unreleased lease, which it had not tracked, and trusts it for another 120 seconds. A learner waiting in close() for its final round reaches its 270-second timeout and gives up; the round commits seconds later.](../assets/pythia-shutdown.svg)

The fixes made shutdown and takeover account for what the other nodes had already observed. `close()` now releases the lease even when it must cancel a publish that exceeds its timeout. Followers track leases on every poll and retain those observations when they take over, so an old lease does not receive a fresh waiting period. Communication failures log their tracebacks without stopping lease renewal, and a failed round is reconstructed from the mailbox before retrying. Regression tests cover these failure cases.

Upload logs now include bytes and elapsed time, and round logs identify participants and commits, along with ownership changes and retries. A bandwidth preflight checks the connection before training starts: each node times a 26 MB PUT and refuses to join below 2 MB/s. The next run could then begin with machines whose uploads met that requirement.

## Keeping the faster nodes busy

Those machines passed the upload check, but one consistently trained at about 230,000 tokens per second while the others reached 270,000 to 285,000. The difference was roughly 18%. With a fixed quarter of every batch assigned to each node, the faster three would have finished 28 minutes earlier, leaving the slower node to process its last 740 steps alone.

The remaining data would still belong to that node even though the others had capacity to process it. To make that work available to them, the training plan was divided into units that any node could reserve. Each unit covers one of the four batch slices over 20 consecutive dataset steps, giving 800 units for the full run.

A node reserves a pending unit through the bucket, loads its batches and returns for another unit when its data reader needs more work. Faster nodes therefore take more units, while a slower node takes fewer. The units retain their reference dataset indices and learning rates; their assignment to machines is decided during execution. The [Shared Queue](../patterns/00-shared-queue.md) pattern describes this mechanism and its API.

![Four nodes process the same 24 units, four slices of six groups of dataset steps. A processes a unit in one time unit, B and C in two, D in four. With a fixed split each node owns one slice of every step: A finishes its six units at t = 6, B and C at t = 12, and D works alone until t = 24. With a shared queue each node claims the next unclaimed unit when it is ready: A processes 11 units, B and C 5 each and D 3, and all are done at t = 12.](../assets/take-a-number.svg)

At a progress check 420 steps into the run, the nodes had claimed 23, 22, 23 and 19 units. Their work assignments were already differing as they returned to the queue at different rates.

## Completing the run

The full run began with preflight upload rates of 6.4, 16.6, 8.9 and 19.4 MB/s. It took 2 h 14 min from the first unit to the release of the last lease, processing about 8.4 billion tokens in 16,000 local optimizer steps. At the listed rental price, that is about US$1.50 of GPU time, plus a few minutes for bootstrap.

The work was distributed as follows:

| Node | Units | Local optimizer steps |
|---|---|---|
| 0 | 215 | 4300 |
| 1 | 194 | 3880 |
| 2 | 216 | 4320 |
| 3 | 175 | 3500 |

All 348 rounds committed: 337 with four participants, 10 with three and one with a single participant. No snapshot was rejected, and no poll or publish failed. The only two takeovers occurred at the end, each within seconds of the previous closer releasing its lease.

The shared queue reduced the imbalance, but the final reserved units could still leave one node working alone. Round 346 included all four nodes; round 347 contained only the last node's remaining updates.

## Comparing the trained model

The checkpoints were scored on the 1,024 sequences at dataset step 140,000, using mean cross-entropy. Neither the main reference run nor this run had encountered that batch during its first 4,000 dataset steps.

At the end, the distributed model's loss was 3.634, compared with 3.781 for the official step-4,000 checkpoint. Its previous all-node round scored 3.618: the final single-node round increased loss by 0.016. Keeping the last updates therefore made the measured result slightly worse.

The intermediate evaluations show how the runs reached those values. They were performed on an Apple GPU during training, while the final evaluation ran on a 3090 afterwards; the two evaluation paths differ in the third decimal. A distributed checkpoint is labeled by the dataset frontier reached by its committed rounds, so some rows compare nearby rather than identical steps.

| Dataset step (run / reference) | Official loss | This run's loss (round) |
|---|---|---|
| 0 | 11.006 | |
| 64 | 10.324 | 10.303 (7) |
| ~480 / 512 | 6.184 | 6.346 (38) |
| 1000 | 5.151 | |
| ~1228 | | 4.641 (103) |
| ~1601 | | 4.120 (135) |
| ~1845 / 2000 | 4.115 | 3.977 (156) |
| ~2114 | | 3.881 (179) |
| ~2364 | | 3.815 (200) |
| ~2613 | | 3.763 (222) |
| ~2866 | | 3.723 (244) |
| 3000 | 3.878 | |
| ~3125 | | 3.690 (266) |
| ~3373 | | 3.666 (288) |
| ~3629 | | 3.641 (310) |
| ~3880 | | 3.623 (332) |
| 4000 | 3.781 | 3.618 (346), 3.634 (347) |

![Loss against the dataset step, from 0 to 4,000. The training loss of the four nodes, smoothed, falls from 11.0 together, to about 3.65 at step 4,000. The official Pythia-14M checkpoints, scored on a held-out batch, go from 11.006 at step 0 to 6.184 at step 512, 5.151 at 1,000, 4.115 at 2,000, 3.878 at 3,000 and 3.781 at 4,000. This run's committed rounds, on the same batch, score 10.303 at step 64, 6.346 at about 480, then below the official curve from about step 1,200, down to 3.618 at step 4,000. Nine PolyPythia seeds score 3.732 to 3.762 at step 4,000.](../assets/pythia-14m-loss.svg)

At step 64, the nine reference seeds span 10.277 to 10.354, around the official loss of 10.324. This run's 10.303 falls within that range. The next comparison has the distributed run at about step 480 and the reference at step 512, with losses of 6.346 and 6.184 respectively.

Later, the distributed run reaches the official step-3,000 loss at about dataset step 2,100 and the official step-4,000 loss at about step 2,600.

At step 4,000, `compare` scores the official checkpoint at 3.781. The nine additional seeds have a mean of 3.741 and standard deviation of 0.009, with losses from 3.732 to 3.762. The official checkpoint has the highest loss among those ten reference runs.

The distributed checkpoints are 0.123 and 0.107 below that mean, before and after the final single-node round respectively. Both remain below the reference range on this evaluation batch.

## What might explain the loss difference

The lower loss does not identify which part of the training procedure caused the difference. Although the data and initial weights match, the distributed run performs 16,000 local AdamW updates across four replicas, using 256 sequences per update. The reference performs 4,000 updates using 1,024 sequences each.

The smaller local batches could explain some of the loss difference. DiLoCo's outer Nesterov momentum could also contribute. This experiment does not measure those effects separately; two follow-up runs would help distinguish them:

1. A single node processes all four slices of each dataset step sequentially, taking four updates of 256 sequences without merging. A similar curve would support the smaller-batch explanation.
2. The four-node run uses outer learning rate 1 and no outer momentum, averaging parameters at each round. Comparing it with this run would measure the effect of the configured outer optimizer.

## Limits

This experiment covers a 14-million-parameter model and the first 4,000 dataset steps. Its communication pattern is dominated by two embedding fragments, and it does not establish how the approach scales to larger models. The PolyPythias extend to 410M parameters, providing larger reference models with additional seeds for comparison.

The smoke tests exercised failures caused by a slow connection and shutdown, but node churn and nodes joining mid-run were not systematically tested. Evaluation is also limited to loss on one fixed batch. Scoring these checkpoints with the task harness used by Pythia would test whether the loss difference carries over to downstream tasks.
