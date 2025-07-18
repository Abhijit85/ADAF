# amaf/agents/__init__.py
from .base import Agent
from ..core import InputData, AgentOutput
from .tabu_synth import TabuSynthAgent
from .contextron import ContextronAgent
from .visura import VisuraAgent
from .summa_craft import SummaCraftAgent
from .trend_analyser import TrendAnalyserAgent      #  ← add
from .calculator import CalculatorAgent
from .post_check import PostCheckAgent
# from .topk_filter import TopKFilterAgent         #  ← if you use it
from .mcp_controller import MCPController          #  ← if you created MCP file
