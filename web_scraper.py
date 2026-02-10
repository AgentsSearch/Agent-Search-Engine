"""
MCP Registry Web Scraper
Fetches AI agent metadata from MCP registry and converts to unified schema format.
"""

import requests
import json
from typing import List, Dict, Optional
from datetime import datetime
import hashlib
from urllib.parse import urljoin

number_of_agents = 10


class MCPRegistryScraper:
    """
    Scraper for MCP (Model Context Protocol) Registry to fetch AI agent metadata.
    Converts MCP schema to unified agent format for indexing.
    """
    
    def __init__(self, base_url: str = "https://registry.mcp.run"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'MCP-Agent-Search-Engine/1.0',
            'Accept': 'application/json'
        })
    
    def fetch_agent_list(self) -> List[Dict]:
        """
        Fetch the list of available agents from the MCP registry.
        
        Returns:
            List of agent metadata dictionaries
        """
        try:
            # Change value of 'number_of_agents' to get first n number of agents (alphabetical order)
            endpoint = f'https://registry.modelcontextprotocol.io/v0.1/servers?limit={number_of_agents}'
            
            try:
                print(f"  Trying: {endpoint}")
                response = self.session.get(endpoint, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    print(f"  ✓ Successfully connected to: {endpoint}")
                    
                    # Handle different response structures
                    if isinstance(data, list):
                        return data
                    elif isinstance(data, dict):
                        # Try common keys
                        for key in ['agents', 'servers', 'data', 'items']:
                            if key in data:
                                return data[key]
                        return [data]  # Single agent
                    
            except requests.exceptions.RequestException as e:
                print(f"  ✗ Failed: {e}")
            
            print("⚠ Could not find API endpoint, trying to scrape HTML...")
            return 0
            # return self._scrape_html_listing()
            
        except Exception as e:
            print(f"Error fetching agent list: {e}")
            return []
    
    def _scrape_html_listing(self) -> List[Dict]:
        """
        Fallback: Scrape HTML page if API is not available.
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            print("BeautifulSoup not installed. Install with: pip install beautifulsoup4 --break-system-packages")
            return []
        
        try:
            response = self.session.get(self.base_url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            agents = []
            # Look for common patterns in agent listings
            for card in soup.find_all(['div', 'article', 'li'], class_=lambda x: x and any(
                term in str(x).lower() for term in ['agent', 'card', 'server', 'item']
            )):
                agent_data = self._extract_agent_from_html(card)
                if agent_data:
                    agents.append(agent_data)
            
            print(f"Found {len(agents)} agents from HTML scraping")
            return agents
            
        except Exception as e:
            print(f"Error scraping HTML: {e}")
            return []
    
    def _extract_agent_from_html(self, element) -> Optional[Dict]:
        """Extract agent data from HTML element."""
        try:
            # Template extraction - adjust selectors based on actual HTML
            name_elem = element.find(['h2', 'h3', 'h4', 'a', 'strong'])
            name = name_elem.get_text(strip=True) if name_elem else "Unknown"
            
            description_elem = element.find(['p', 'div', 'span'], class_=lambda x: x and 'description' in str(x).lower())
            description = description_elem.get_text(strip=True) if description_elem else ""
            
            link = element.find('a')
            url = link['href'] if link and link.get('href') else ""
            
            return {
                'name': name,
                'description': description,
                'url': urljoin(self.base_url, url) if url else "",
                'source': 'mcp_html'
            }
        except:
            return None
    
    def fetch_agent_details(self, agent_identifier: str) -> Dict:
        """
        Fetch detailed information for a specific agent.
        
        Args:
            agent_identifier: Agent name, ID, or URL path
            
        Returns:
            Detailed agent metadata
        """
        try:
            # Try different endpoint patterns
            endpoints = [
                f"{self.base_url}/api/agents/{agent_identifier}",
                f"{self.base_url}/api/v1/agents/{agent_identifier}",
                f"{self.base_url}/api/servers/{agent_identifier}",
            ]
            
            for endpoint in endpoints:
                try:
                    response = self.session.get(endpoint, timeout=10)
                    if response.status_code == 200:
                        return response.json()
                except:
                    continue
                    
            return {}
                
        except Exception as e:
            print(f"  Error fetching details: {e}")
            return {}
    
    def convert_to_unified_schema(self, mcp_agent: Dict) -> Dict:
        """
        Convert MCP schema to unified agent schema.
        
        Args:
            mcp_agent: Agent data in MCP format
            
        Returns:
            Agent data in unified schema format
        """
        # Generate unique agent_id from source and name
        source_name = mcp_agent.get('name', 'unknown')
        agent_id = hashlib.md5(f"mcp_{source_name}".encode()).hexdigest()[:16]
        mcp_agent_meta = mcp_agent.get('_meta')
        mcp_agent = mcp_agent.get('server')
        
        print(mcp_agent)
        # Extract source URL
        source_url = (
            mcp_agent.get('url') or 
            mcp_agent.get('websiteUrl') or
            mcp_agent.get('repository', {}).get('url') or 
            mcp_agent.get('remotes', [{}])[0].get('url') or 
            ''
        )
        
        unified = {
            # Identity
            'agent_id': agent_id,
            'name': mcp_agent.get('name', 'Unknown'),
            'source': 'mcp',
            'source_url': source_url,
            
            # Description
            'description': mcp_agent.get('description', ''),
            'short_description': (mcp_agent.get('description', '')[:200] + '...') if len(mcp_agent.get('description', '')) > 200 else mcp_agent.get('description', ''),
            
            # Capabilities
            'tools': mcp_agent.get('tools', []),
            'detected_capabilities': self._extract_capabilities(mcp_agent),
            'llm_backbone': mcp_agent.get('framework') or mcp_agent.get('llm_backbone') or 'Unknown',
            
            # Evaluation Data
            'arena_elo': mcp_agent.get('arena_elo'),
            'arena_battles': mcp_agent.get('arena_battles'),
            'community_rating': mcp_agent.get('rating'),
            'rating_count': mcp_agent.get('rating_count', 0),
            
            # Metadata
            'pricing': mcp_agent.get('pricing', 'unknown'),
            'last_updated': mcp_agent_meta.get('io.modelcontextprotocol.registry/official').get('updatedAt', 'Unknown'),
            'indexed_at': datetime.utcnow().isoformat(),
            
            # Computed
            'description_embedding': None,
            'testability_tier': 'n/a',
            
            # Raw data
            '_raw_mcp_data': mcp_agent
        }
        
        return unified
    
    def _extract_capabilities(self, mcp_agent: Dict) -> List[str]:
        """Extract structured capability list from MCP agent data."""
        capabilities = []
        
        # Direct capabilities field
        if 'capabilities' in mcp_agent:
            caps = mcp_agent['capabilities']
            if isinstance(caps, list):
                capabilities.extend([str(c) for c in caps])
            elif isinstance(caps, dict):
                capabilities.extend([str(k) for k in caps.keys()])
        
        # Infer from tools
        if 'tools' in mcp_agent:
            for tool in mcp_agent['tools']:
                if isinstance(tool, dict):
                    tool_name = tool.get('name') or tool.get('type') or str(tool)
                    capabilities.append(f"tool:{tool_name}")
                elif isinstance(tool, str):
                    capabilities.append(f"tool:{tool}")
        
        # Extract from description
        desc = mcp_agent.get('description', '').lower()
        capability_keywords = ['search', 'generate', 'analyze', 'process', 'create', 'manage', 'monitor']
        for keyword in capability_keywords:
            if keyword in desc:
                capabilities.append(keyword)
        
        return list(set(capabilities))
    
    def fetch_documentation(self, agent: Dict) -> Dict[str, str]:
        """
        Fetch documentation for an agent.
        
        Args:
            agent: Agent metadata (raw or unified)
            
        Returns:
            Dictionary with documentation sources
        """
        docs = {}
        
        # Build list of possible README URLs
        readme_urls = []
        
        # Check for explicit documentation URLs
        for key in ['readme_url', 'documentation_url', 'docs_url']:
            if key in agent and agent[key]:
                readme_urls.append(agent[key])
        
        # Try to construct GitHub README URL if repository is available
        repo_url = agent.get('source_url') or agent.get('websiteUrl', '')
        if 'github.com' in repo_url:
            # Clean up URL
            repo_url = repo_url.rstrip('/')
            readme_urls.append(f"{repo_url}/raw/main/README.md")
            readme_urls.append(f"{repo_url}/raw/master/README.md")
        
        # Fetch README
        for url in readme_urls:
            if not url:
                continue
            try:
                response = self.session.get(url, timeout=10)
                if response.status_code == 200:
                    docs['readme'] = response.text
                    print(f"    ✓ Fetched README from {url}")
                    break
            except Exception as e:
                print(f"    ✗ Failed to fetch from {url}: {e}")
                continue
        
        return docs
    
    def scrape_all_agents(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Main method: Scrape all agents from registry.
        
        Args:
            limit: Maximum number of agents to fetch
            
        Returns:
            List of agents in unified schema format
        """
        print("🔍 Fetching agent list from MCP registry...")
        agent_list = self.fetch_agent_list()
        
        if not agent_list:
            print("❌ No agents found. The registry might be unavailable or using a different structure.")
            return []
        
        if limit:
            agent_list = agent_list[:limit]
        
        print(f"📊 Found {len(agent_list)} agents to process")
        
        unified_agents = []
        
        for i, agent_summary in enumerate(agent_list, 1):
            agent_name = agent_summary.get('name', 'Unknown')
            print(f"\n[{i}/{len(agent_list)}] Processing: {agent_name}")
            
            # Fetch detailed information if identifier available
            agent_id = agent_summary.get('id') or agent_summary.get('name')
            if agent_id:
                print(f"  📡 Fetching detailed info...")
                detailed_agent = self.fetch_agent_details(agent_id)
                if detailed_agent:
                    agent_summary.update(detailed_agent)
            
            # Convert to unified schema
            unified_agent = self.convert_to_unified_schema(agent_summary)
            
            # Fetch documentation
            print(f"  📄 Fetching documentation...")
            docs = self.fetch_documentation(unified_agent)
            if docs:
                print(docs)
            unified_agent['documentation'] = docs
            
            unified_agents.append(unified_agent)
            
            # Rate limiting - be respectful
            # time.sleep(1)
        
        print(f"\n✅ Successfully scraped {len(unified_agents)} agents")
        return unified_agents
    
    def save_to_file(self, agents: List[Dict], filename: str = "mcp_agents.json"):
        """Save scraped agents to JSON file."""
        filepath = f"./{filename}"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(agents, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved to {filepath}")
        return filepath


class DocumentationProcessor:
    """
    Processes documentation for semantic chunking and embedding.
    Implements the pipeline from section 1.7.
    """
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_documentation(self, text: str) -> List[Dict]:
        """
        Chunk documentation semantically with overlap.
        """
        # Token approximation: ~4 chars per token
        char_chunk_size = self.chunk_size * 4
        char_overlap = self.overlap * 4
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + char_chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                sentence_end = max(
                    text.rfind('. ', start, end),
                    text.rfind('.\n', start, end),
                    text.rfind('!\n', start, end),
                    text.rfind('?\n', start, end)
                )
                if sentence_end > start:
                    end = sentence_end + 1
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    'chunk_id': chunk_id,
                    'text': chunk_text,
                    'start_pos': start,
                    'end_pos': end,
                    'token_count_estimate': len(chunk_text) // 4
                })
                chunk_id += 1
            
            start = end - char_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def extract_capabilities_llm_placeholder(self, text: str) -> Dict:
        """
        Placeholder for LLM extraction of capabilities, limitations, requirements.
        Replace with actual LLM API call in production.
        """
        capabilities = []
        limitations = []
        requirements = []

        pass
    
    def calculate_quality_score(self, doc: Dict) -> float:
        """
        Calculate documentation quality score (0-1).
        Might replace with anotehr LLM API call.
        """
        score = 0.0

        return score
    
    def process_agent_documentation(self, agent: Dict) -> Dict:
        """Full documentation processing pipeline."""
        docs = agent.get('documentation', {})
        
        all_chunks = []
        
        # Process README
        if 'readme' in docs and docs['readme']:
            readme_chunks = self.chunk_documentation(docs['readme'])
            for chunk in readme_chunks:
                chunk['source_type'] = 'readme'
                chunk['agent_id'] = agent['agent_id']
            all_chunks.extend(readme_chunks)
        
        # Extract capabilities
        if docs and docs.get('readme'):
            extracted = self.extract_capabilities_llm_placeholder(docs['readme'])
            agent['llm_extracted'] = extracted
        else:
            agent['llm_extracted'] = {'capabilities': [], 'limitations': [], 'requirements': []}
        
        # Calculate quality score
        agent['documentation_quality'] = self.calculate_quality_score(docs)
        
        # Add chunks
        agent['documentation_chunks'] = all_chunks
        
        print(f"  📝 Created {len(all_chunks)} documentation chunks")
        print(f"  ⭐ Quality score: {agent['documentation_quality']:.2f}")
        
        return agent


def main():
    """Main execution function."""
    print("="*70)
    print("MCP REGISTRY WEB SCRAPER")
    print("="*70)
    
    # Configuration
    LIMIT = 10  # Number of agents to scrape (set to None for all)
    
    # Step 1: Scrape agents
    print("\n" + "="*70)
    print("STEP 1: SCRAPING AGENTS FROM MCP REGISTRY")
    print("="*70)
    
    scraper = MCPRegistryScraper()
    agents = scraper.scrape_all_agents(limit=LIMIT)
    
    if not agents:
        print("\n❌ No agents were scraped. Exiting.")
        return
    
    # Step 2: Process documentation
    print("\n" + "="*70)
    print("STEP 2: PROCESSING DOCUMENTATION")
    print("="*70)
    
    processor = DocumentationProcessor(chunk_size=512, overlap=50)
    
    processed_agents = []
    for agent in agents:
        print(f"\n📦 Processing documentation for: {agent['name']}")
        processed_agent = processor.process_agent_documentation(agent)
        processed_agents.append(processed_agent)
    
    # Step 3: Save results
    print("\n" + "="*70)
    print("STEP 3: SAVING RESULTS")
    print("="*70)
    
    output_file = scraper.save_to_file(processed_agents, "mcp_agents.json")
    
    # Summary
    print("\n" + "="*70)
    print("SCRAPING SUMMARY")
    print("="*70)
    print(f"✅ Total agents processed: {len(processed_agents)}")
    print(f"✅ Total documentation chunks: {sum(len(a.get('documentation_chunks', [])) for a in processed_agents)}")
    print(f"✅ Average quality score: {sum(a.get('documentation_quality', 0) for a in processed_agents) / len(processed_agents):.2f}")
    print(f"\n📁 Output file: {output_file}")
    
    # Show sample
    print("\n" + "="*70)
    print("SAMPLE AGENT DATA")
    print("="*70)
    if processed_agents:
        sample = processed_agents[0]
        print(f"\n📦 {sample['name']}")
        print(f"   ID: {sample['agent_id']}")
        print(f"   Source: {sample['source_url']}")
        print(f"   Description: {sample['short_description']}")
        print(f"   Capabilities: {', '.join(sample['detected_capabilities'][:5])}")
        print(f"   Tools: {len(sample['tools'])}")
        print(f"   Documentation chunks: {len(sample.get('documentation_chunks', []))}")
        print(f"   Quality score: {sample.get('documentation_quality', 0):.2f}")


if __name__ == "__main__":
    main()