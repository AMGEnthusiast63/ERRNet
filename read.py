from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

event_file = "/data/zhuchenyao/dip/ERRNet/checkpoints/errnet/logs/Jun02_12-30-12_nvidia/events.out.tfevents.1780403412.nvidia"

ea = EventAccumulator(event_file)
ea.Reload()

print("Tags:")
print(ea.Tags())

for tag in ea.Tags()['scalars']:
    print("\n", tag)
    for event in ea.Scalars(tag)[:20]:
        print(event.step, event.value)