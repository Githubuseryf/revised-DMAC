def _default_env(name):
    if name == "nous":
        from dmac.tool.envs.nous import NousToolEnv
        return NousToolEnv
    elif name == "retool":
        from dmac.tool.envs.retool import ReToolEnv
        return ReToolEnv
    else:
        raise NotImplementedError(f"Tool environment {name} is not implemented")