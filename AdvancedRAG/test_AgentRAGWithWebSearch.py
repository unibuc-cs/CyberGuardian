import CyberguardianAgentWithRagAndWebSearch
from pprint import pprint
from CyberguardianAgentWithRagAndWebSearch import CyberguardianAgentWithRagAndWebSearch
if __name__ == "__main__":
    # Load the model
    # Initialize the agent
    agent = CyberguardianAgentWithRagAndWebSearch(model=None, tokenizer=None, retriever=None)

    res = agent.run_agent(question="What are the types of agent memory?")
    pprint(f"Result of the agent: {res}")
