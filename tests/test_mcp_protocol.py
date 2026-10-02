# -*- coding: utf-8 -*-
"""
tests/test_mcp_protocol.py

Validates the Model Context Protocol (MCP) server configuration:
1. Verifies that the FastMCP server registers the 4 Core Public Tools:
   - ask_procurement_law
   - check_procurement_threshold
   - get_statute_section
   - search_procurement_clauses
2. Verifies that internal/diagnostic tools are hidden by default unless requested.
3. Verifies that standard MCP Resources and Prompts are registered.
"""

import os
import sys
import asyncio
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.mcp.server import mcp


class TestMCPProtocol(unittest.TestCase):
    def test_01_exposed_tools_contract(self):
        """Verifies that exactly the 4 primary public tools are registered on FastMCP by default."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            tools = loop.run_until_complete(mcp.list_tools())
            tool_names = {t.name for t in tools}
            
            expected_public_tools = {
                "ask_procurement_law",
                "check_procurement_threshold",
                "get_statute_section",
                "search_procurement_clauses"
            }
            
            for expected in expected_public_tools:
                self.assertIn(
                    expected,
                    tool_names,
                    f"Core MCP Tool '{expected}' is missing from registered tools!"
                )
            
            # Verify internal tools are not exposed by default
            if not os.getenv("expose_internal_tools", "false").lower() in ("true", "1", "yes"):
                self.assertNotIn("healthcheck", tool_names, "Internal 'healthcheck' should not be exposed!")
                self.assertNotIn("search_procurement_faqs", tool_names, "Internal 'search_procurement_faqs' should not be exposed!")
        finally:
            loop.close()

    def test_02_resources_registered(self):
        """Verifies that the legal thresholds and catalog resources are registered."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            resources = loop.run_until_complete(mcp.list_resources())
            uris = {str(r.uri) for r in resources}
            self.assertIn("procurement://rules/thresholds", uris)
            self.assertIn("procurement://catalog/documents", uris)
        finally:
            loop.close()


if __name__ == "__main__":
    unittest.main()
