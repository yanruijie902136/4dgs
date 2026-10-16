# Optional SIBR viewer

The Python adapter speaks the original SIBR remote Gaussian viewer protocol.
Install the viewer separately using the [upstream viewer instructions](https://github.com/graphdeco-inria/gaussian-splatting#interactive-viewers).

Enable the listener explicitly when starting training:

```bash
4dgs train experiment=dnerf/bouncingballs data.root=./datasets/bouncingballs \
  run.output_dir=./outputs/bouncingballs viewer.enabled=true viewer.port=6017
```

Connect `SIBR_remoteGaussian_app` to port 6017. The default bind address is
`127.0.0.1`; set `viewer.host` if necessary. For a remote training machine,
forward that port through SSH or your editor's port-forwarding feature.

The viewer can request coarse/fine renders and pause training while connected.
Camera time follows the dataset video trajectory. Connections are owned by the
viewer adapter and closed when training exits. Network throughput affects the
interactive frame rate; this is separate from measured CUDA render timings.
