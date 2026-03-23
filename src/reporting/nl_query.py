"""
Natural Language Query Interface.
Parses queries like "Show ships in zone B from last 10 minutes".
"""
import re, time
from typing import List, Dict, Optional

class NLQueryParser:
    def __init__(self):
        self.patterns = {
            "zone_query": re.compile(r"ships?\s+(?:in|inside|within)\s+(?:zone\s+)?(.+?)(?:\s+(?:from|in)\s+(?:last|past)\s+(\d+)\s+(min|hour|sec))?$", re.I),
            "dark_vessel": re.compile(r"dark\s+vessels?", re.I),
            "threat_above": re.compile(r"threat\s+(?:score\s+)?(?:above|over|>)\s+(\d+)", re.I),
            "threat_level": re.compile(r"(high|medium|low)\s+threat", re.I),
            "time_range": re.compile(r"(?:last|past)\s+(\d+)\s+(min|hour|sec)", re.I),
            "ship_type": re.compile(r"(cargo|tanker|fishing|military)\s+ships?", re.I),
            "ship_count": re.compile(r"(?:how many|count|total)\s+(?:ships?|vessels?)", re.I),
            "fleet": re.compile(r"(?:fleet|formation|convoy)s?", re.I),
            "summary": re.compile(r"(?:summary|overview|report|status)", re.I),
        }

    def parse(self, query: str) -> Dict:
        query = query.strip()
        for qtype, pattern in self.patterns.items():
            match = pattern.search(query)
            if match:
                result = {"type": qtype}
                if qtype == "zone_query":
                    result["zone_name"] = match.group(1).strip()
                    if match.group(2):
                        result["time_seconds"] = self._parse_time(int(match.group(2)), match.group(3))
                elif qtype == "threat_above":
                    result["threshold"] = int(match.group(1))
                elif qtype == "threat_level":
                    result["level"] = match.group(1).upper()
                elif qtype == "time_range":
                    result["time_seconds"] = self._parse_time(int(match.group(1)), match.group(2))
                elif qtype == "ship_type":
                    result["ship_type"] = match.group(1).lower()
                return result
        return {"type": "unknown", "original_query": query}

    def _parse_time(self, amount, unit):
        u = unit.lower()
        if u.startswith("min"): return amount * 60
        if u.startswith("hour"): return amount * 3600
        return amount


class QueryExecutor:
    def __init__(self):
        self.parser = NLQueryParser()

    def execute(self, query, detections=None, alerts=None, tracks=None, fleets=None, timestamp=None):
        parsed = self.parser.parse(query)
        ts = timestamp or time.time()
        detections = detections or []
        alerts = alerts or []
        qtype = parsed.get("type", "unknown")

        if qtype == "dark_vessel":
            r = [d for d in detections if d.get("is_dark_vessel")]
            return {"results": r, "count": len(r), "message": f"🔴 {len(r)} dark vessel(s) detected."}
        elif qtype == "threat_above":
            th = parsed.get("threshold", 50)
            r = [d for d in detections if d.get("threat_score", 0) > th]
            return {"results": r, "count": len(r), "message": f"⚠️ {len(r)} ships above threat {th}."}
        elif qtype == "threat_level":
            lv = parsed.get("level", "HIGH")
            r = [d for d in detections if d.get("threat_level", "").upper() == lv]
            return {"results": r, "count": len(r), "message": f"🎯 {len(r)} {lv} threat ships."}
        elif qtype == "ship_type":
            st = parsed.get("ship_type", "")
            r = [d for d in detections if d.get("ship_type", "").lower() == st]
            return {"results": r, "count": len(r), "message": f"🚢 {len(r)} {st} vessels."}
        elif qtype == "ship_count":
            return {"results": detections, "count": len(detections), "message": f"📋 {len(detections)} ships detected."}
        elif qtype == "zone_query":
            zn = parsed.get("zone_name", "")
            r = [a for a in alerts if zn.lower() in a.get("zone_name", "").lower()]
            tr = parsed.get("time_seconds")
            if tr: r = [a for a in r if a.get("timestamp", 0) >= ts - tr]
            return {"results": r, "count": len(r), "message": f"🗺️ {len(r)} alerts in '{zn}'."}
        elif qtype == "fleet":
            fc = len(fleets) if fleets else 0
            return {"results": [], "count": fc, "message": f"⚓ {fc} fleet formation(s)."}
        elif qtype == "summary":
            dc = sum(1 for d in detections if d.get("is_dark_vessel"))
            ht = sum(1 for d in detections if d.get("threat_level") == "HIGH")
            return {"results": {}, "count": len(detections),
                    "message": f"📊 Ships: {len(detections)} | Dark: {dc} | High threat: {ht} | Alerts: {len(alerts)}"}
        return {"results": [], "count": 0, "message": f"❓ Couldn't parse: '{query}'"}
