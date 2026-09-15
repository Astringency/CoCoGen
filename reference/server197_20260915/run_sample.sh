#!/bin/bash

# darcy
# python sample.py --pde_name "darcy" --logdir "./output/samples/" --name "both" --config "./configs/darcy.yaml" --ckpt "./output/withcontrol/2025-11-27T21-17-46_cocogen4darcy/checkpoints/last.ckpt" --sensor "./configs/sample/darcy_both.yaml"
# python sample.py --pde_name "darcy" --logdir "./output/samples/" --name "forward" --config "./configs/darcy.yaml" --ckpt "./output/withcontrol/2025-11-27T21-17-46_cocogen4darcy/checkpoints/last.ckpt" --sensor "./configs/sample/darcy_forward.yaml"
# python sample.py --pde_name "darcy" --logdir "./output/samples/" --name "inverse" --config "./configs/darcy.yaml" --ckpt "./output/withcontrol/2025-11-27T21-17-46_cocogen4darcy/checkpoints/last.ckpt" --sensor "./configs/sample/darcy_inverse.yaml"

# poisson
python sample.py --pde_name "poisson" --logdir "./output/samples/" --name "both" --config "./configs/poisson.yaml" --ckpt "./output/withcontrol/2025-11-27T21-31-52_cocogen4poisson/checkpoints/last.ckpt" --sensor "./configs/sample/poisson_both.yaml"
# python sample.py --pde_name "poisson" --logdir "./output/samples/" --name "forward" --config "./configs/poisson.yaml" --ckpt "./output/withcontrol/2025-11-27T21-31-52_cocogen4poisson/checkpoints/last.ckpt" --sensor "./configs/sample/poisson_forward.yaml"
# python sample.py --pde_name "poisson" --logdir "./output/samples/" --name "inverse" --config "./configs/poisson.yaml" --ckpt "./output/withcontrol/2025-11-27T21-31-52_cocogen4poisson/checkpoints/last.ckpt" --sensor "./configs/sample/poisson_inverse.yaml"

# helmholtz
# python sample.py --pde_name "helmholtz" --logdir "./output/samples/" --name "both" --config "./configs/helmholtz.yaml" --ckpt "./output/withcontrol/2025-11-28T09-24-03_cocogen4helmholtz/checkpoints/last.ckpt" --sensor "./configs/sample/helmholtz_both.yaml"
# python sample.py --pde_name "helmholtz" --logdir "./output/samples/" --name "forward" --config "./configs/helmholtz.yaml" --ckpt "./output/withcontrol/2025-11-28T09-24-03_cocogen4helmholtz/checkpoints/last.ckpt" --sensor "./configs/sample/helmholtz_forward.yaml"
# python sample.py --pde_name "helmholtz" --logdir "./output/samples/" --name "inverse" --config "./configs/helmholtz.yaml" --ckpt "./output/withcontrol/2025-11-28T09-24-03_cocogen4helmholtz/checkpoints/last.ckpt" --sensor "./configs/sample/helmholtz_inverse.yaml"

# ns
# python sample.py --pde_name "nsnonbounded" --logdir "./output/samples/" --name "both" --config "./configs/nsnonbounded.yaml" --ckpt "./output/withcontrol/2025-11-27T21-32-06_cocogen4nsnonbounded/checkpoints/last.ckpt" --sensor "./configs/sample/nsnonbounded_both.yaml"
# python sample.py --pde_name "nsnonbounded" --logdir "./output/samples/" --name "forward" --config "./configs/nsnonbounded.yaml" --ckpt "./output/withcontrol/2025-11-27T21-32-06_cocogen4nsnonbounded/checkpoints/last.ckpt" --sensor "./configs/sample/nsnonbounded_forward.yaml"
# python sample.py --pde_name "nsnonbounded" --logdir "./output/samples/" --name "inverse" --config "./configs/nsnonbounded.yaml" --ckpt "./output/withcontrol/2025-11-27T21-32-06_cocogen4nsnonbounded/checkpoints/last.ckpt" --sensor "./configs/sample/nsnonbounded_inverse.yaml"
