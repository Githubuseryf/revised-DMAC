def _default_tool(name):
    print("***********name*************:",name)
    if name == "prompt":
        from dmac.tool.tools.LLM_tool_dynamic_agent import LLM_Tool_DynamicAgent
        return LLM_Tool_DynamicAgent()
    else:
        raise NotImplementedError(f"Tool {name} not implemented")
