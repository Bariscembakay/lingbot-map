import wandb
api = wandb.Api()
for r in api.runs("bbakay22-ko-university/spatial_memory"):
    if r.name.startswith("scenes96_96f_b4_write4_read2_lingbothead") and not r.id.endswith("-m"):
        print(f"deleting {r.id} state={r.state}")
        r.delete()
print("done")
