import json

import chainlit as cl
import requests
from chainlit.mcp import McpConnection
from dotenv import dotenv_values, load_dotenv
from mcp import ClientSession

load_dotenv()

env = dotenv_values()

chainlit_host = env.get("CHAINLIT_HOST")
chainlit_port = env.get("CHAINLIT_PORT")
mcp_client_entry = env.get("MCP_CLIENT_ENTRY")
mindsdb_sse_url = env.get("MINDSDB_SSE_URL")

assert mcp_client_entry is not None
assert chainlit_host is not None
assert chainlit_port is not None

chainlit_url = f"http://{chainlit_host}:{chainlit_port}"
session_id: str | None = None


mindsdb_server = {"name": "MindsDB"}


# === Life Cycle === #
@cl.on_chat_start
async def on_chat_start():
    global session_id
    session_id = cl.user_session.get("id")

    async_conect_to_mindsdb_mcp = cl.make_async(connect_to_mindsdb_mcp)
    await async_conect_to_mindsdb_mcp()

    # mcp_tools = cl.user_session.get("mcp_tools")
    #
    # if mcp_tools is not None:
    #     await cl.Message(content=mcp_tools).send()


@cl.on_message
async def on_message(msg: cl.Message):
    mcp_tools = cl.user_session.get("mcp_tools")

    if mcp_tools is not None and mcp_tools.get(mindsdb_server.get("name")) is not None:
        mcp_session, _ = cl.context.session.mcp_sessions.get(mindsdb_server.get("name"))  # pyright: ignore
        result = await mcp_session.call_tool(
            "query",
            {
                "query": f"select completion from llama3_model where text = '{msg.content}'"
            },
        )

        content = result.model_dump()["content"][0]["text"]
        content = json.loads(content)["data"][0][0]

        await cl.Message(content=content).send()


@cl.on_stop
async def on_stop():
    await cl.Message(content="Task stopped.").send()


@cl.on_chat_end
async def on_chat_end():
    await cl.Message(content="Session end.").send()


# @cl.on_chat_resume
# async def on_chat_resume(thread: ThreadDict):
#     await cl.Message(content="Session resume").send()


# ====== #


# === MCP === #
@cl.on_mcp_connect  # pyright: ignore[reportArgumentType]
async def on_mcp_connect(connection: McpConnection, session: ClientSession):
    """Called when an MCP connection is established"""

    # List available tools
    result = await session.list_tools()

    # Process tool metadata
    tools = [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.inputSchema,
        }
        for t in result.tools
    ]

    # Store tools for later use
    mcp_tools = cl.user_session.get("mcp_tools", {})

    if mcp_tools is not None:
        mcp_tools[connection.name] = tools

    cl.user_session.set("mcp_tools", mcp_tools)


# @cl.on_mcp_disconnect  # pyright: ignore[reportArgumentType]
# async def on_mcp_disconnect(name: str, session: ClientSession):
#     """Called when an MCP connection is terminated"""


# === Other task === #
def connect_to_mindsdb_mcp():
    assert mindsdb_sse_url is not None

    res = requests.post(
        f"{chainlit_url}/mcp",
        json={
            "clientType": "sse",
            "sessionId": session_id,
            "name": mindsdb_server.get("name"),
            "url": mindsdb_sse_url,
        },
    )

    return res.json()
