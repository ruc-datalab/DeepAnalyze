AGENT_GENERATOR_REGISTRY = {
    "skyagent.agents.oh_codeact.OHCodeActAgent": "skyagent.agents.base.AgentRunner",
    "skyagent.agents.react.ReActAgent": "skyagent.agents.base.AgentRunner",
    "skyagent.agents.react.DummyReactAgent": "skyagent.agents.base.AgentRunner",
}

AGENT_TRAJECTORY_REGISTRY = {
    "skyagent.agents.oh_codeact.OHCodeActAgent": "skyagent.agents.oh_codeact.CodeActTrajectory",
    "skyagent.agents.react.ReActAgent": "skyagent.agents.react.ReaActTrajectory",
    "skyagent.agents.react.DummyReactAgent": "skyagent.agents.react.ReaActTrajectory",
}
