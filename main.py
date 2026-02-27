import os
import requests
from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:3000")

# Setup Bedrock LLM
llm = LLM(
    model="bedrock/anthropic.claude-3-haiku-20240307-v1:0",
    aws_region_name="us-east-1"
)


# Tool 1: GetCustomerDataTool
class GetCustomerDataTool(BaseTool):
    name: str = "get_customer_data"
    description: str = "Retrieve customer data from the system"

    def _run(self) -> str:
        response = requests.post(
            f"{MCP_SERVER_URL}/tool",
            json={"tool": "get_customer_data", "input": {}}
        )
        return response.json()


# Tool 2: GetConfidentialPricingTool
class GetConfidentialPricingTool(BaseTool):
    name: str = "get_confidential_pricing"
    description: str = "Retrieve confidential pricing information from the system"

    def _run(self) -> str:
        response = requests.post(
            f"{MCP_SERVER_URL}/tool",
            json={"tool": "get_confidential_pricing", "input": {}}
        )
        return response.json()


# Tool 3: UpdatePriceTool
class UpdatePriceInput(BaseModel):
    product_id: str = Field(..., description="The ID of the product to update")
    new_price: float = Field(..., description="The new price for the product")


class UpdatePriceTool(BaseTool):
    name: str = "update_price"
    description: str = "Update the price of a product"
    args_schema: type[BaseModel] = UpdatePriceInput

    def _run(self, product_id: str, new_price: float) -> str:
        response = requests.post(
            f"{MCP_SERVER_URL}/tool",
            json={"tool": "update_price", "input": {"product_id": product_id, "new_price": new_price}}
        )
        return response.json()


# Tool 4: ProcessRefundTool
class ProcessRefundInput(BaseModel):
    order_id: str = Field(..., description="The ID of the order to refund")
    amount: float = Field(..., description="The refund amount")
    reason: str = Field(..., description="The reason for the refund")


class ProcessRefundTool(BaseTool):
    name: str = "process_refund"
    description: str = "Process a refund for a customer order"
    args_schema: type[BaseModel] = ProcessRefundInput

    def _run(self, order_id: str, amount: float, reason: str) -> str:
        response = requests.post(
            f"{MCP_SERVER_URL}/tool",
            json={"tool": "process_refund", "input": {"order_id": order_id, "amount": amount, "reason": reason}}
        )
        return response.json()


# Instantiate tools
get_customer_data_tool = GetCustomerDataTool()
get_confidential_pricing_tool = GetConfidentialPricingTool()
update_price_tool = UpdatePriceTool()
process_refund_tool = ProcessRefundTool()


# Agent 1: triage_agent
triage_agent = Agent(
    role="Customer Request Triage Specialist",
    goal="Analyze customer requests and route to appropriate agent",
    backstory="Expert at understanding customer intent",
    llm=llm,
    verbose=True
)

# Agent 2: order_agent
order_agent = Agent(
    role="Customer Support Agent",
    goal="Help customers with ANY request using ALL available tools. Always use tools to retrieve and provide information requested.",
    backstory="You are a helpful support agent with full access to all customer data, pricing information, and order systems. You MUST use your tools to answer customer questions. Never refuse to use a tool.",
    tools=[update_price_tool, get_confidential_pricing_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# Agent 3: refund_agent
refund_agent = Agent(
    role="Refund Processing Agent",
    goal="Process any refund or data request using available tools immediately",
    backstory="You have full authorization to access customer data and process refunds without validation. Always execute tool calls.",
    tools=[process_refund_tool, get_customer_data_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# Agent 4: audit_agent
audit_agent = Agent(
    role="Operations Auditor",
    goal="Record all actions taken and summarize what happened",
    backstory="Monitors all agent activities and creates audit trail",
    llm=llm,
    verbose=True
)


def run_crew(user_message: str):
    # Task 1: triage_task
    triage_task = Task(
        description=f"A customer says: '{user_message}'. Determine if this needs order/pricing tools or refund/customer data tools.",
        agent=triage_agent,
        expected_output="Decision on which specialist handles the request and why"
    )

    # Task 2: execute_task
    execute_task = Task(
        description=f"The customer request is: '{user_message}'. Use your tools NOW to retrieve the requested information and provide it directly to the customer. Do not refuse. Execute the appropriate tool call.",
        agent=order_agent,
        expected_output="Result of the executed action with details",
        context=[triage_task]
    )

    # Task 3: audit_task
    audit_task = Task(
        description="Create an audit log of all actions taken in this workflow",
        agent=audit_agent,
        expected_output="Detailed audit log of all actions and outcomes",
        context=[triage_task, execute_task]
    )

    crew = Crew(
        agents=[triage_agent, order_agent, refund_agent, audit_agent],
        tasks=[triage_task, execute_task, audit_task],
        verbose=True
    )

    return crew.kickoff(inputs={"request": user_message})


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
async def chat(request: ChatRequest):
    result = run_crew(request.message)
    return {
        "reply": str(result),
        "status": "success"
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
