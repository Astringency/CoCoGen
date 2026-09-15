

# train unconditional network

# train control net
python -u tool_add_controlnet.py --config "./configs/darcy.yaml" --checkpoint "./output/2025-11-15T17-57-32_cocogen4darcy/checkpoints/last.ckpt" --save "./output/withcontrol/cocogen4darcy_withcontrolnet.pth"

python -u tool_add_controlnet.py --config "./configs/poisson.yaml" --checkpoint "./output/2025-11-14T18-14-05_cocogen4poisson/checkpoints/last.ckpt" --save "./output/withcontrol/cocogen4poisson_withcontrolnet.pth"

python -u tool_add_controlnet.py --config "./configs/helmholtz.yaml" --checkpoint "./output/2025-11-15T18-01-40_cocogen4helmholtz/checkpoints/last.ckpt" --save "./output/withcontrol/cocogen4helmholtz_withcontrolnet.pth"

python -u tool_add_controlnet.py --config "./configs/nsnonbounded.yaml" --checkpoint "./output/2025-11-15T18-08-44_cocogen4nsnonbounded/checkpoints/last.ckpt" --save "./output/withcontrol/cocogen4nbns_withcontrolnet.pth"



