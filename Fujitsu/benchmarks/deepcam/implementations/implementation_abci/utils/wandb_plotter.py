import wandb
import pandas as pd
import math
import sys

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


sns.set_theme()
sns.set_style("whitegrid")

api = wandb.Api()

# run is specified by <entity>/<project>/<run_id>
runs = {}
runs["20210916-JOB_7940609-256_nodes-local_shuffling"] = "1,024 GPUs Local"
runs["20211014-JOB_8132761-256_nodes-partial_0.25_shuffling-rand_pairs-LR-0.0055"] = "1,024 GPUs Partial (0.25)"
runs["20210916-JOB_7940702-256_nodes-partial_0.5_shuffling"] = "1,024 GPUs Partial (0.5)"
runs["20210916-JOB_7945044-256_nodes-partial_0.9_shuffling"] = "1,024 GPUs Partial (0.9)"

runs["20210929-JOB_8013067-512_nodes-local_shuffling-LR-0.011"] = "2,048 GPUs Local"
runs["20211010-JOB_8104674-512_nodes-local_shuffling-LR-0.011"] = "2,048 GPUs Local"
runs["20210928-JOB_8011474-512_nodes-partial_0.9_shuffling-rand_pairs-LR-0.011"] = "2,048 GPUs Partial (0.9)"
runs["20211011-JOB_8108511-512_nodes-partial_0.9_shuffling-rand_pairs-LR-0.011"] = "2,048 GPUs Partial (0.9)"


frames = []
for conf, label in runs.items():
	print("Processing: " + label)
	run = api.run("bgerofi/DeepCAM-ABCI_GC/" + conf).history()
	run["Configuration"] = label

	prev_epoch = None
	prev_ts = None
	prev_iter = None
	perf_dict = {"GPUs" : [], "Duration": [], "Shuffling": []}
	for index, row in run.iterrows():
		if row["_step"] % 10 != 0:
			continue

		_iter = row["_step"]
		epoch = row["epoch"]
		ts = row["_timestamp"]

		if epoch == prev_epoch and _iter == prev_iter + 10:
			#print(epoch, ts - prev_ts)
			duration = ts - prev_ts
			if label[0:5] == "1,024":
				duration *= (float(54272)*2/2048/10)
			else:
				duration *= (float(54272)*2/4096/10)

			perf_dict["GPUs"] += [label[0:5]]
			perf_dict["Shuffling"] += [label[11:]]
			perf_dict["Duration"] += [duration]

		prev_iter = _iter
		prev_epoch = epoch
		prev_ts = ts

	frames.append(pd.DataFrame(perf_dict))

df = pd.concat(frames, ignore_index=True)
print(df)

plt.figure(figsize=(5,5))
ax = sns.barplot(x="GPUs", y="Duration", hue="Shuffling", data=df)
ax.set_ylabel('Training time per epoch (s)', size=15, fontdict=dict(weight='bold'))
ax.set_xlabel('Number of GPUs', size=15, fontdict=dict(weight='bold'))
plt.tight_layout()
plt.ylim(0, 300)
#plt.xlim(0, 29.5)
plt.setp(ax.get_legend().get_title(), fontsize='12')
plt.savefig("deepcam-perf.pdf")

#sys.exit(0)


frames = []

for conf, label in runs.items():
	print("Processing: " + label)
	run = api.run("bgerofi/DeepCAM-ABCI_GC/" + conf).history()
	run["Configuration"] = label
	run = run[['epoch', 'eval_accuracy', 'Configuration']]
	if conf == "20210916-JOB_7940609-256_nodes-local_shuffling":
		last = {"epoch" : 29, "eval_accuracy": 0.791725, "Configuration": label}
		run = run.append(last, ignore_index=True)
	run = run.dropna()
	print(run)

	frames.append(run)

df = pd.concat(frames, ignore_index=True)
#print(df)
#print("duplicated:")
#print(df[df.index.duplicated()])

plt.figure(figsize=(5,5))
plt.xticks(size=12)
plt.yticks(size=12)
#plt.rcParams["figure.figsize"] = (9, 4)

ax = sns.lineplot(x="epoch", y="eval_accuracy", data=df, hue="Configuration", style="Configuration", markers=True, dashes=True, ci=None);
#ax = sns.lineplot(x="epoch", y="eval_accuracy", data=df, hue="Configuration");
ax.set_ylabel('Validation accuracy', size=15, fontdict=dict(weight='bold'))
ax.set_xlabel('Epoch number', size=15, fontdict=dict(weight='bold'))
plt.tight_layout()
plt.ylim(0.25, 0.83)
plt.xlim(0, 29.5)
params = {'legend.fontsize': 16}
plt.rcParams.update(params)
#plt.setp(ax.get_legend().get_texts(), fontsize='8') # for legend text
#plt.setp(ax.get_legend().get_title(), fontsize='12')
#plt.rc('legend', fontsize=16)

plt.savefig("deepcam-accuracy.pdf")


