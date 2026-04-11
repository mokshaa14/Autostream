"""
tools/lead_capture.py
Mock lead capture tool for AutoStream agent.
In production, this would POST to a CRM API (HubSpot, Salesforce, etc.)
"""

import json
from datetime import datetime


def mock_lead_capture(name: str, email: str, platform: str) -> dict:
    """
    Simulates capturing a qualified lead into a CRM system.
    
    Args:
        name     : Full name of the prospect
        email    : Email address of the prospect
        platform : Creator platform (YouTube, Instagram, TikTok, etc.)
    
    Returns:
        dict with status and lead_id
    """
    # Simulate a lead ID (in production this would come from the CRM)
    lead_id = f"LEAD-{int(datetime.now().timestamp())}"

    lead_data = {
        "lead_id": lead_id,
        "name": name,
        "email": email,
        "platform": platform,
        "captured_at": datetime.now().isoformat(),
        "source": "Inflx Social-to-Lead Agent",
        "product": "AutoStream",
        "status": "new"
    }

    # Simulate writing to a local leads log file
    try:
        with open("leads_log.json", "a") as f:
            f.write(json.dumps(lead_data) + "\n")
    except Exception:
        pass  # Non-critical for demo

    print("\n" + "=" * 55)
    print("✅  LEAD CAPTURED SUCCESSFULLY")
    print("=" * 55)
    print(f"  Lead ID  : {lead_id}")
    print(f"  Name     : {name}")
    print(f"  Email    : {email}")
    print(f"  Platform : {platform}")
    print(f"  Time     : {lead_data['captured_at']}")
    print("=" * 55 + "\n")

    return {
        "success": True,
        "lead_id": lead_id,
        "message": f"Lead captured successfully: {name}, {email}, {platform}"
    }
