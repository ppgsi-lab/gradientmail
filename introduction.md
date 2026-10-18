---
title: Introduction
---

# Introduction

{% capture readme %}{% include_relative README.md %}{% endcapture %}
{{ readme | remove_first: "# GradientMail" | replace: "/README.md)", "/)" | replace: ".md)", ".html)" }}
