from __future__ import annotations

from dataclasses import dataclass, field
import re


@dataclass
class HiringProfile:
    role_title: str = ""
    seniority: str = ""

    min_experience: float | None = None
    max_experience: float | None = None

    # What the company fundamentally wants
    required_capabilities: list[str] = field(default_factory=list)
    preferred_capabilities: list[str] = field(default_factory=list)

    # Explicit technologies (only evidence)
    required_skills: list[str] = field(default_factory=list)
    preferred_skills: list[str] = field(default_factory=list)

    # Things to penalize
    disqualifiers: list[str] = field(default_factory=list)

    # Behavioral expectations
    required_behaviors: list[str] = field(default_factory=list)

    # Culture
    company_preferences: list[str] = field(default_factory=list)

    # Logistics
    preferred_locations: list[str] = field(default_factory=list)

    # Derived recruiter intent
    target_domains: list[str] = field(default_factory=list)
    preferred_company_types: list[str] = field(default_factory=list)

    requires_recent_hands_on: bool = False
    requires_production_experience: bool = False
    requires_product_company: bool = False
    prefers_startup_background: bool = False
    requires_retrieval_experience: bool = False
    requires_ranking_experience: bool = False
    requires_evaluation_experience: bool = False

    raw_text: str = ""

class HiringProfileBuilder:

    EXPERIENCE_PATTERN = re.compile(
        r"(\d+)\s*[-–]\s*(\d+)\s*years?",
        re.IGNORECASE
    )

    BULLET = re.compile(r"^\s*[-•]\s*(.+)$")

    def __init__(self):
        pass

    def build(self, jd_text: str) -> HiringProfile:

        profile = HiringProfile()
        profile.raw_text = jd_text
        lines = [l.strip() for l in jd_text.splitlines() if l.strip()]

        # ---------------------------------------------------------
        # Role title
        # ---------------------------------------------------------

        if lines:
            profile.role_title = lines[0]

        # ---------------------------------------------------------
        # Experience
        # ---------------------------------------------------------

        exp = self.EXPERIENCE_PATTERN.search(jd_text)

        if exp:
            profile.min_experience = float(exp.group(1))
            profile.max_experience = float(exp.group(2))

            if profile.max_experience <= 3:
                profile.seniority = "junior"
            elif profile.max_experience <= 6:
                profile.seniority = "mid"
            elif profile.max_experience <= 10:
                profile.seniority = "senior"
            else:
                profile.seniority = "staff+"

        # ---------------------------------------------------------
        # Parse sections
        #
        # PHASE 1 FIX: the real job_description.txt (and most JDs written
        # in prose rather than a bulleted slide deck) has one requirement
        # per line with NO leading "-"/"•" marker at all -- the previous
        # BULLET regex (`^\s*[-•]\s*(.+)$`) never matched a single line of
        # the actual target document, so required_capabilities,
        # preferred_capabilities, disqualifiers, and required_behaviors
        # were *always* empty lists in production, even though the unit
        # tests (which use synthetic "- bullet" fixtures) made this look
        # like it worked. HiringProfile cannot be the single source of
        # truth for recruiter intent if it silently extracts nothing from
        # the real JD.
        #
        # Fix: use the document's actual structure -- sections are
        # delimited by blank lines, and each line inside a triggered
        # section is one requirement/behavior, whether or not it happens
        # to start with a bullet character. A blank-line paragraph break
        # closes the current section unless the next paragraph's first
        # line is itself a recognized section header (this is what
        # correctly separates "Things we explicitly do NOT want" from the
        # unrelated "On location, comp, and logistics" paragraph that
        # follows it in the same block-adjacent text, and closes "vibe
        # check" before "How to read between the lines").
        # ---------------------------------------------------------

        blocks = [b for b in re.split(r"\n\s*\n", jd_text) if b.strip()]

        section_targets = {
            "required": profile.required_capabilities,
            "preferred": profile.preferred_capabilities,
            "avoid": profile.disqualifiers,
            "culture": profile.required_behaviors,
        }

        for block in blocks:
            current = None  # each new paragraph block starts with no open section
            for raw_line in block.splitlines():
                line = raw_line.strip()
                if not line:
                    continue

                lower = line.lower()

                if "things you absolutely need" in lower:
                    current = "required"
                    continue
                if "things we'd like" in lower:
                    current = "preferred"
                    continue
                if "things we explicitly do not want" in lower:
                    current = "avoid"
                    continue
                if "vibe check" in lower:
                    current = "culture"
                    continue

                if current is None:
                    continue

                m = self.BULLET.match(line)
                item = m.group(1).strip() if m else line
                section_targets[current].append(item)

        text = jd_text.lower()

        if "production" in text:
            profile.requires_production_experience = True

        if "product company" in text:
            profile.requires_product_company = True

        if "startup" in text or "series a" in text:
            profile.prefers_startup_background = True

        if "retrieval" in text:
            profile.requires_retrieval_experience = True

        if "ranking" in text:
            profile.requires_ranking_experience = True

        if "evaluation" in text or "ndcg" in text or "a/b" in text:
            profile.requires_evaluation_experience = True

        if "writes code" in text or "production code" in text:
            profile.requires_recent_hands_on = True
        if "product company" in text:
            profile.preferred_company_types.append("product")

        if "startup" in text:
            profile.preferred_company_types.append("startup")
        if "retrieval" in text:
            profile.target_domains.append("information_retrieval")

        if "ranking" in text:
            profile.target_domains.append("ranking")

        if "recommendation" in text:
            profile.target_domains.append("recommendation")

        if "search" in text:
            profile.target_domains.append("search")

        if "llm" in text:
            profile.target_domains.append("llm")
        # ---------------------------------------------------------
        # Locations
        # ---------------------------------------------------------

        known_locations = [
            "Pune",
            "Noida",
            "Delhi",
            "Delhi NCR",
            "Mumbai",
            "Hyderabad",
            "Bangalore",
            "Bengaluru",
            "India",
        ]

        for loc in known_locations:
            if loc.lower() in jd_text.lower():
                profile.preferred_locations.append(loc)

        profile.preferred_locations = sorted(
            list(set(profile.preferred_locations))
        )

        return profile