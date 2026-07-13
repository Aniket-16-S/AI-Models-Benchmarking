"""Generate the deterministic 60-case meeting-extraction benchmark.

Run: python generate_test_dataset.py
The committed JSON is generated from 12 scenarios x 5 entity variants so that
the benchmark is large enough for meaningful aggregate metrics without hiding
ground truth in hand-edited, inconsistent records.
"""

import json
from pathlib import Path


VARIANTS = [
    {"project": "Atlas", "owner": "Aniket", "day": "Tuesday", "service": "Postgres", "contact": "Maya"},
    {"project": "Beacon", "owner": "Priya", "day": "Wednesday", "service": "Redis", "contact": "Omar"},
    {"project": "Cedar", "owner": "Luis", "day": "Thursday", "service": "Kafka", "contact": "Nina"},
    {"project": "Delta", "owner": "Chen", "day": "Friday", "service": "MongoDB", "contact": "Iris"},
    {"project": "Ember", "owner": "Fatima", "day": "Monday", "service": "Elasticsearch", "contact": "Ravi"},
]


SCENARIOS = [
    {
        "key": "migration",
        "lines": [
            "Sarah: Let’s review the {project} migration status.",
            "{owner}: The application work is nearly done.",
            "{owner}: {service} connection pooling drops under peak load.",
            "Sarah: Is that blocking the deployment?",
            "{owner}: Yes, it could cause failed requests in production.",
            "Sarah: What is the next action?",
            "{owner}: I will patch the pool configuration by {day}.",
            "Sarah: Please add a load-test result to the release ticket.",
            "{owner}: I will attach it after the configuration change.",
            "Sarah: We will not deploy until the test is green.",
            "{owner}: Understood; I own both follow-ups.",
        ],
        "tasks": ["Patch {service} connection pool configuration by {day} (Assignee: {owner})", "Attach load-test result to the release ticket after the configuration change (Assignee: {owner})"],
        "risks": ["Production request failures caused by dropped {service} connection pooling"],
    },
    {
        "key": "vendor",
        "lines": [
            "Maya: The {project} vendor contract is still unsigned.",
            "{owner}: Legal returned two requested edits.",
            "Maya: Does the vendor know our launch date?",
            "{owner}: Not yet; they are waiting for the revised agreement.",
            "Maya: Delayed signature could postpone onboarding.",
            "{owner}: I will send the revised contract to {contact} by {day}.",
            "Maya: Ask for a signature timeline in the same email.",
            "{owner}: I will record their response in procurement.",
            "Maya: Finance approval is already complete.",
            "{owner}: Good, so legal is the only open dependency.",
            "Maya: Let’s review it in the next stand-up.",
        ],
        "tasks": ["Send revised contract to {contact} by {day} (Assignee: {owner})", "Record {contact}'s signature-timeline response in procurement (Assignee: {owner})"],
        "risks": ["Vendor onboarding delay due to an unsigned contract"],
    },
    {
        "key": "incident",
        "lines": [
            "Iris: Yesterday’s {project} outage lasted eighteen minutes.",
            "{owner}: The primary symptom was elevated {service} timeouts.",
            "Iris: Have we confirmed a root cause?",
            "{owner}: No, the logs rotate before the full trace is retained.",
            "Iris: That makes the next incident harder to diagnose.",
            "{owner}: I will extend log retention to thirty days by {day}.",
            "Iris: Please also write the incident timeline.",
            "{owner}: I will publish it in the operations channel.",
            "Iris: Customer communications are handled separately.",
            "{owner}: I will link the timeline to the incident ticket.",
            "Iris: Thank you; we need evidence before closing it.",
        ],
        "tasks": ["Extend {service} log retention to thirty days by {day} (Assignee: {owner})", "Publish the incident timeline in the operations channel (Assignee: {owner})", "Link the incident timeline to the incident ticket (Assignee: {owner})"],
        "risks": ["Future incident root-cause analysis is impaired by short log retention"],
    },
    {
        "key": "security",
        "lines": [
            "Ravi: The {project} security scan found a critical library vulnerability.",
            "{owner}: It is in the {service} client package.",
            "Ravi: Is there a patched release available?",
            "{owner}: Yes, but we have not tested compatibility.",
            "Ravi: Leaving it unpatched exposes the service to a known exploit.",
            "{owner}: I will upgrade the package in staging by {day}.",
            "Ravi: Run the regression suite before merging.",
            "{owner}: I will post the test outcome to the security ticket.",
            "Ravi: Production rollout needs my approval.",
            "{owner}: I will request it after staging passes.",
            "Ravi: Keep the scan finding open until then.",
        ],
        "tasks": ["Upgrade the {service} client package in staging by {day} (Assignee: {owner})", "Post regression-test outcome to the security ticket (Assignee: {owner})", "Request Ravi's production-rollout approval after staging passes (Assignee: {owner})"],
        "risks": ["Known exploit exposure from an unpatched {service} client vulnerability"],
    },
    {
        "key": "data_quality",
        "lines": [
            "Nina: The {project} dashboard totals do not match finance.",
            "{owner}: Duplicate events are arriving from the {service} pipeline.",
            "Nina: Are executives using this dashboard?",
            "{owner}: Yes, it feeds the weekly revenue review.",
            "Nina: Incorrect totals could lead to a bad revenue decision.",
            "{owner}: I will add a deduplication rule by {day}.",
            "Nina: Please backfill the affected reporting period too.",
            "{owner}: I will validate the corrected total with {contact}.",
            "Nina: Mark the dashboard as provisional until validation.",
            "{owner}: I will add that notice today.",
            "Nina: Good; do not remove it early.",
        ],
        "tasks": ["Add a deduplication rule to the {service} pipeline by {day} (Assignee: {owner})", "Backfill the affected reporting period (Assignee: {owner})", "Validate corrected dashboard totals with {contact} (Assignee: {owner})", "Mark the dashboard as provisional today (Assignee: {owner})"],
        "risks": ["Incorrect revenue decisions caused by duplicate-event dashboard totals"],
    },
    {
        "key": "launch",
        "lines": [
            "Omar: The {project} launch checklist is almost complete.",
            "{owner}: The only missing item is the {service} rollback procedure.",
            "Omar: Can the team deploy without it?",
            "{owner}: We could, but recovery would be slow if the release fails.",
            "Omar: That is an unacceptable launch risk.",
            "{owner}: I will draft the rollback procedure by {day}.",
            "Omar: Walk the on-call team through it afterward.",
            "{owner}: I will schedule the walkthrough with {contact}.",
            "Omar: Capture the dry-run result in the checklist.",
            "{owner}: I will update it after the walkthrough.",
            "Omar: Launch remains conditional on a successful dry run.",
        ],
        "tasks": ["Draft the {service} rollback procedure by {day} (Assignee: {owner})", "Schedule rollback walkthrough with {contact} (Assignee: {owner})", "Update the launch checklist with the dry-run result (Assignee: {owner})"],
        "risks": ["Slow recovery from a failed release without a {service} rollback procedure"],
    },
    {
        "key": "hiring",
        "lines": [
            "Chen: The {project} team still has an open data-engineer role.",
            "{owner}: We have screened six candidates.",
            "Chen: Why has no interview panel been scheduled?",
            "{owner}: The interview rubric has not been approved.",
            "Chen: The vacancy may delay the {service} migration.",
            "{owner}: I will send the rubric to {contact} for approval by {day}.",
            "Chen: Book panels as soon as it is approved.",
            "{owner}: I will coordinate available interviewers.",
            "Chen: Recruiting will continue sourcing in parallel.",
            "{owner}: I will update the hiring tracker.",
            "Chen: Escalate if approval slips past the date.",
        ],
        "tasks": ["Send interview rubric to {contact} for approval by {day} (Assignee: {owner})", "Coordinate interviewer availability after rubric approval (Assignee: {owner})", "Update the hiring tracker (Assignee: {owner})"],
        "risks": ["{service} migration delay caused by the unfilled data-engineer role"],
    },
    {
        "key": "compliance",
        "lines": [
            "Fatima: The {project} audit evidence is incomplete.",
            "{owner}: We are missing access-review records for {service}.",
            "Fatima: When is the external audit?",
            "{owner}: It begins next week.",
            "Fatima: Missing records could produce an audit finding.",
            "{owner}: I will export the access-review records by {day}.",
            "Fatima: Have {contact} verify the export is complete.",
            "{owner}: I will request that verification immediately.",
            "Fatima: Store the final evidence in the audit folder.",
            "{owner}: I will upload it after verification.",
            "Fatima: Please notify me once it is available.",
        ],
        "tasks": ["Export {service} access-review records by {day} (Assignee: {owner})", "Request export-completeness verification from {contact} (Assignee: {owner})", "Upload final access-review evidence to the audit folder after verification (Assignee: {owner})", "Notify Fatima when the audit evidence is available (Assignee: {owner})"],
        "risks": ["External audit finding caused by missing {service} access-review records"],
    },
    {
        "key": "performance",
        "lines": [
            "Luis: The {project} API response time increased after the last release.",
            "{owner}: The {service} query is consuming most of the request time.",
            "Luis: Is the customer-facing endpoint affected?",
            "{owner}: Yes, users see delays during peak traffic.",
            "Luis: Continued delays may increase checkout abandonment.",
            "{owner}: I will profile the query by {day}.",
            "Luis: Bring a proposed index to the performance review.",
            "{owner}: I will document the benchmark before and after the change.",
            "Luis: Do not ship an index without the benchmark.",
            "{owner}: Agreed; I will link it to the change request.",
            "Luis: We need a measurable improvement.",
        ],
        "tasks": ["Profile the {service} query by {day} (Assignee: {owner})", "Propose an index at the performance review (Assignee: {owner})", "Document before-and-after query benchmarks (Assignee: {owner})", "Link the benchmark to the change request (Assignee: {owner})"],
        "risks": ["Checkout abandonment caused by peak-traffic API delays"],
    },
    {
        "key": "integration",
        "lines": [
            "Priya: The {project} partner integration is returning incomplete orders.",
            "{owner}: The {service} mapping omits the tax field.",
            "Priya: Has the partner confirmed their payload format?",
            "{owner}: Not yet; their documentation conflicts with the sandbox.",
            "Priya: Incomplete orders could create billing discrepancies.",
            "{owner}: I will send payload samples to {contact} by {day}.",
            "Priya: Ask them to confirm the canonical field mapping.",
            "{owner}: I will update our mapper after their response.",
            "Priya: Add an automated contract test as well.",
            "{owner}: I will create it before enabling production traffic.",
            "Priya: Keep the feature flag off until then.",
        ],
        "tasks": ["Send payload samples to {contact} by {day} (Assignee: {owner})", "Request confirmation of the canonical field mapping from {contact} (Assignee: {owner})", "Update the {service} mapper after {contact}'s response (Assignee: {owner})", "Create an automated integration contract test before enabling production traffic (Assignee: {owner})"],
        "risks": ["Billing discrepancies caused by incomplete partner orders"],
    },
    {
        "key": "backup",
        "lines": [
            "Aniket: The {project} disaster-recovery drill exposed a backup gap.",
            "{owner}: The {service} restore took longer than the recovery target.",
            "Aniket: What caused the delay?",
            "{owner}: The backup runbook lacks the new encryption step.",
            "Aniket: That could extend downtime during a real incident.",
            "{owner}: I will update the runbook by {day}.",
            "Aniket: Repeat the restore drill using the revised steps.",
            "{owner}: I will invite {contact} to observe the drill.",
            "Aniket: Record the achieved recovery time.",
            "{owner}: I will add it to the resilience report.",
            "Aniket: We need proof that the target is met.",
        ],
        "tasks": ["Update the {service} backup runbook by {day} (Assignee: {owner})", "Repeat the restore drill using the revised runbook (Assignee: {owner})", "Invite {contact} to observe the restore drill (Assignee: {owner})", "Add achieved recovery time to the resilience report (Assignee: {owner})"],
        "risks": ["Extended incident downtime because the {service} runbook lacks the encryption step"],
    },
    {
        "key": "budget",
        "lines": [
            "Sarah: The {project} cloud spend is above the approved forecast.",
            "{owner}: Unused {service} capacity is running overnight.",
            "Sarah: Do we know the monthly impact?",
            "{owner}: Finance estimates it at twelve percent of the service budget.",
            "Sarah: Continued overspend could force cuts to planned work.",
            "{owner}: I will identify idle capacity by {day}.",
            "Sarah: Submit a rightsizing plan to {contact}.",
            "{owner}: I will include projected savings in that plan.",
            "Sarah: Finance needs the plan before the next forecast cycle.",
            "{owner}: I will track the approved changes.",
            "Sarah: Please report the first month of savings.",
        ],
        "tasks": ["Identify idle {service} capacity by {day} (Assignee: {owner})", "Submit a rightsizing plan with projected savings to {contact} (Assignee: {owner})", "Track approved rightsizing changes (Assignee: {owner})", "Report first-month savings (Assignee: {owner})"],
        "risks": ["Planned-work cuts caused by continued cloud overspend"],
    },
]


def render(value: str | list[str], variant: dict[str, str]) -> str | list[str]:
    if isinstance(value, list):
        return [item.format(**variant) for item in value]
    return value.format(**variant)


def build_dataset() -> list[dict]:
    records = []
    for scenario in SCENARIOS:
        for number, variant in enumerate(VARIANTS, start=1):
            records.append(
                {
                    "id": f"meet_{scenario['key']}_{number:03d}",
                    "transcript": "\n".join(render(scenario["lines"], variant)),
                    "ground_truth": {
                        "tasks": render(scenario["tasks"], variant),
                        "risks": render(scenario["risks"], variant),
                    },
                }
            )
    return records


if __name__ == "__main__":
    output = Path(__file__).with_name("test_dataset.json")
    dataset = build_dataset()
    output.write_text(json.dumps(dataset, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(dataset)} records to {output}")
