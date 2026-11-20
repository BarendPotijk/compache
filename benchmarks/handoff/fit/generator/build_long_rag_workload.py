#!/usr/bin/env python3
"""Build a deterministic long-form, full-history RAG workload."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "examples" / "rag_long"
BASE = datetime(2026, 4, 6, 9, tzinfo=timezone.utc)
SEED = 20261116
COMPANIES = {
    "alder": {
        "name": "Alder & Finch",
        "cadence": 9,
        "region": "Rotterdam",
        "base": 28,
        "budget": 420,
        "standard": 5,
        "express": 2,
        "leave": 26,
    },
    "briar": {
        "name": "Briar Works",
        "cadence": 23,
        "region": "Utrecht",
        "base": 35,
        "budget": 560,
        "standard": 7,
        "express": 3,
        "leave": 24,
    },
    "cinder": {
        "name": "Cinder & Co.",
        "cadence": 47,
        "region": "Eindhoven",
        "base": 21,
        "budget": 310,
        "standard": 4,
        "express": 1,
        "leave": 29,
    },
}
# Frozen split: every profile occurs in training; the held-out conversation uses a
# known profile, distinct case register values, and a later date.
THREADS = [(f"long-{cid}-{n}", cid, "train") for cid in COMPANIES for n in (1, 2)] + [
    ("long-alder-heldout", "alder", "heldout")
]
TURN_COUNT = 72


def iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def digest(value: str, n: int = 16) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:n]


def citation(part: dict) -> dict[str, str]:
    return {"part_id": part["part_id"], "quote": part["text"]}


def compact(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def docs(cid: str) -> dict[str, tuple[str, str, str]]:
    c = COMPANIES[cid]
    n = c["name"]
    data = {
        "returns_old": (
            f"{n} accepts unused goods within {c['base']} days after delivery. Proof of purchase is required; clearance goods are excluded. Items must be in original packaging.",
            "returns",
            "v1",
        ),
        "returns": (
            f"Effective 1 March 2026, {n} accepts unused goods within {c['base'] + 7} days after delivery. Proof of purchase is required; clearance goods are excluded; items must be in original packaging.",
            "returns",
            "v2",
        ),
        "standard": (
            f"For in-stock orders, {n} standard delivery usually takes {c['standard']} business days after dispatch. The estimate excludes public holidays; a tracking email is sent when the parcel leaves the warehouse.",
            "delivery",
            "v1",
        ),
        "express": (
            f"For in-stock orders, {n} express delivery usually takes {c['express']} business days after dispatch and is available only in mainland Netherlands. The estimate excludes public holidays.",
            "delivery",
            "v1",
        ),
        "delivery": (
            f"For in-stock orders, {n} standard delivery usually takes {c['standard']} business days after dispatch. Express usually takes {c['express']} business days and is available only in mainland Netherlands. Both estimates exclude public holidays; tracking is sent at dispatch.",
            "delivery",
            "v1",
        ),
        "pricing": (
            f"The {n} team plan costs €{c['budget']} monthly for up to 25 seats. Additional seats cost €18 each per month. Annual billing is invoiced at the start of term; there is no setup fee for the standard plan.",
            "pricing",
            "v2",
        ),
        "hosting": (
            f"{n} stores EU production customer records in {c['region']}, Netherlands. Backups are encrypted and retained for 30 days. This residency statement covers production customer records, not employee email or third-party support tickets.",
            "hosting",
            "v2",
        ),
        "people": (
            f"Eligible full-time {n} employees receive {c['leave']} paid leave days per calendar year. Part-time allowance is prorated by contracted hours. Managers must approve requests before booking; unused leave handling is governed by local law.",
            "people",
            "v2",
        ),
        "brand": (
            f"The approved {n} customer motto is ‘Made for everyday journeys.’ Use the wording and punctuation exactly in campaign material. Do not imply a lifetime guarantee; product cover is limited to the written warranty.",
            "brand",
            "v3",
        ),
        "warranty": (
            f"{n} products have a two-year limited written warranty from delivery for manufacturing defects. Accidental damage and normal wear are excluded. Customers should provide order number and photographs when opening a claim; assessment normally takes five business days.",
            "warranty",
            "v1",
        ),
        "support": (
            f"{n} support acknowledges standard cases within one business day, Monday through Friday. Urgent delivery issues should include the order number and dispatch email. Support cannot promise a delivery date beyond the current carrier estimate.",
            "support",
            "v1",
        ),
        "operations": (
            f"{n} onboarding work records the requested outcome, applicable source version, evidence owner, customer-facing scope and next action. Cross-functional handoffs retain their case ID and unresolved questions; an estimate is not a guarantee and an unapproved exception is not policy.",
            "operations",
            "v1",
        ),
    }
    return {
        key: (text, family, version) for key, (text, family, version) in data.items()
    }


def handbook_expansion(key: str, company: str, thread_num: int = 0) -> str:
    """A real handbook section adds procedural context around the compact fact."""
    sections = {
        "returns": [
            "Customer care should first identify the order and delivery date, then confirm that the item is unused. The date on the delivery record starts the customer-facing return window; dispatch date is not substituted for delivery. If a record is incomplete, staff should ask for the order reference instead of estimating eligibility.",
            "The return checklist is applied consistently across web and store-originated orders. Proof of purchase can be an order confirmation or another record that links the item to the transaction. Packaging helps protect the item in transit, but it does not replace the unused condition or proof requirement.",
            "Clearance status is determined from the original sale record. A later price adjustment does not turn a regular item into clearance stock, and a clearance item remains excluded even when the customer has kept its packaging. Staff should explain the exclusion plainly and avoid suggesting that an exception has been approved.",
            "The current policy revision takes precedence for requests received on or after its effective date. Earlier conversations can explain what a customer was previously told, but the current published window controls a new eligibility assessment. The continuing conditions should be checked separately from the changed number of days.",
            "When summarizing the rule, distinguish eligibility from the mechanics of sending an item back. Do not promise a refund date unless the payment workflow provides one. A customer may be directed to the returns portal for a label and status updates after eligibility has been established.",
            "A concise response should state the time limit and the main conditions together. Omitting the delivery-date anchor can make the limit ambiguous. Avoid describing the policy as unconditional, because unused condition, proof, packaging, and the clearance exclusion all affect eligibility.",
            "If a customer asks about a damaged item, route the case to the warranty process rather than treating damage as an ordinary change-of-mind return. The returns rule and defect coverage answer different questions. Staff should not infer that an excluded return has warranty coverage without checking the written warranty.",
            "For internal review, retain the policy version used, the date checked, and the order evidence considered. This supports consistent handling when a customer returns to a case later. It does not authorize a staff member to waive a published condition.",
        ],
        "delivery": [
            "The published delivery durations are usual estimates for in-stock orders, not guaranteed arrival dates. Their clock begins after dispatch, so warehouse preparation time occurs before the estimate starts. A response should keep those qualifiers close to the number rather than presenting a duration as a promise from order placement.",
            "Standard and express services share the same dispatch starting point and public-holiday exclusion. Comparing them is useful only when the order is in stock and the destination is eligible for the chosen service. The estimate difference does not account for a carrier disruption or a late warehouse dispatch.",
            "Express availability is limited to mainland Netherlands. Staff should establish the destination before recommending it; the shorter estimate alone does not establish that the service can be purchased for every address. Standard delivery remains the applicable comparison when express is unavailable.",
            "The tracking email is sent when the parcel leaves the warehouse. Tracking confirms dispatch and gives access to carrier updates, but it does not change the estimate or create a guaranteed delivery date. Customers should use the latest carrier status when asking about an individual parcel.",
            "Business-day estimates exclude public holidays. A weekend or public holiday therefore should not be counted as an ordinary business day when translating the published range into a calendar expectation. Avoid calculating an exact date unless order dispatch and the relevant holiday calendar are both known.",
            "For a speed-sensitive order, explain the faster service and its destination restriction in the same sentence. If the customer is outside the eligible area, do not imply that an exception can be arranged. The carrier estimate remains the best available timing information after dispatch.",
            "An order marked out of stock falls outside the published in-stock estimate. In that case, ask the fulfillment team for availability or replenishment information rather than applying the standard or express duration. A faster transport service cannot make unavailable inventory dispatch sooner.",
            "When a parcel appears delayed, request the order number and dispatch email so support can locate the shipment. Do not reset the estimate based on the time a customer first contacted support. The dispatch event and current carrier scan are the relevant operational references.",
        ],
        "pricing": [
            "The monthly base charge includes the stated seat allowance. To estimate a larger team, add only the seats above that allowance at the published marginal monthly rate. Seat counts should be confirmed for the billing period, since a one-seat error changes the monthly total by the same marginal amount.",
            "Annual billing describes when the term is invoiced, not a discount or a different seat allowance. The invoice is issued at the beginning of the term. Do not infer an annual percentage saving unless a separate offer explicitly states one.",
            "The standard plan has no setup fee under this version of the price sheet. This statement applies to standard-plan onboarding and should not be generalized to custom implementation work or third-party services. Any such work requires its own written quote.",
            "A clear quote separates the base monthly fee, included seats, and extra-seat charge. This lets a customer recalculate the amount if the team size changes. State whether a figure is monthly and whether it assumes the standard plan before comparing it with another offer.",
            "For an arithmetic estimate, first subtract included seats from the requested total, then multiply the remainder by the per-seat rate, and add that amount to the base fee. The calculation is a list-price estimate and does not account for tax or a separately approved contract adjustment.",
            "Billing questions should be checked against the current version because prices and allowances can change. Earlier summaries may accurately describe an old quotation while no longer representing the current published amount. Preserve the version and effective date when handing a billing case to finance.",
            "Unused seats do not automatically create a credit under the price facts in this guide. Do not promise refunds, proration, or mid-term changes unless the signed order form or billing terms establish them. Escalate a disputed invoice with the invoice number and order form.",
            "A customer comparing monthly and annual payment should be told the invoice timing that is actually documented. The plan price remains the listed monthly amount unless a distinct annual price is provided. Keep the answer within the published facts and avoid inventing a discount.",
        ],
        "hosting": [
            "The residency statement concerns production customer records. It does not by itself describe every system that may process operational data, employee information, support correspondence, or telemetry. Keep the stated scope intact when responding to a privacy or procurement questionnaire.",
            "Encrypted backups are retained for thirty days. Backup retention is a recovery control, not a promise that every deleted record remains available to a customer for that period. Requests about deletion, legal holds, or restoration need the applicable data-handling procedure.",
            "A location statement should name the data category and the country together. Saying that all company data is in one location would overstate the evidence. The production customer-record boundary is the documented unit for this statement.",
            "Employee email and third-party support tickets are expressly outside the production customer-record residency claim. Do not answer a broader question about those systems by repeating the customer-record location. Refer the requester to the relevant system owner for separate evidence.",
            "Procurement responses should distinguish storage location from encryption and retention. These are separate controls and answer different questions. The handbook establishes the location, encrypted backup status, and retention duration stated above, but does not specify every key-management or replication detail.",
            "If a customer asks where a particular record resides, first confirm that it is a production customer record. A support attachment or employee message may belong to a different system. Categorize the information before applying the documented residency statement.",
            "The location in this guide is the Netherlands location for the named company. Do not infer that every processor, disaster-recovery copy, or temporary diagnostic artifact uses that location. Such claims require a separate architecture or vendor record.",
            "A concise answer can state the location and immediately note the excluded categories. This prevents a correct narrow statement from being read as a blanket promise about all data handled by the business.",
        ],
        "people": [
            "The annual allowance applies to eligible full-time employees and is counted per calendar year. Part-time employees receive a prorated allowance based on contracted hours. The guide does not provide an individual balance, which should be checked in the leave system.",
            "A request must be approved by a manager before the employee books travel or treats the leave as confirmed. A request submitted in the system is not itself approval. Employees should wait for the recorded decision and coordinate handover with their team.",
            "Unused leave handling is governed by local law. The handbook does not establish a universal carryover or cash-payment rule, so managers should refer questions about a balance at year end to People Operations rather than infer a policy.",
            "For staffing plans, teams should discuss absence early and record coverage for customer-facing duties. Planning guidance does not change the employee's allowance or substitute for approval. Managers remain responsible for confirming that a request follows the normal process.",
            "A response about a specific employee's remaining balance requires the employee's record and cannot be calculated from the annual entitlement alone. Prior leave taken, contract changes, and start date may affect the available balance.",
            "When explaining proration, use contracted hours as the basis. Do not assume a fixed number of days for every part-time schedule unless the relevant hours and calculation method are available. People Operations can confirm an individual result.",
            "The annual number is a paid-leave entitlement, not a guarantee that every requested date will be approved. Managers consider coverage and the applicable process before recording approval. Communicate the entitlement and approval step separately.",
            "People records are sensitive. Share an individual's leave details only through the approved internal channel and with a work-related need. General handbook guidance can be shared without exposing a colleague's personal balance.",
        ],
        "brand": [
            "The approved motto is a fixed customer-facing line. Preserve its words and punctuation when quoting it, including the final period. A paraphrase may communicate the idea but should not be presented as the approved exact motto.",
            "Campaign language must not suggest that every product lasts forever or has lifetime coverage. Product cover is limited to the written warranty. If a draft discusses durability, have the warranty qualifier reviewed before publication.",
            "Brand review checks that the approved line is used in the right context and is not paired with an unsupported product claim. Keep objective product specifications separate from emotional campaign copy, and cite the specification source when one is used.",
            "A local campaign may adapt surrounding copy while retaining the approved motto verbatim. Translation requires the normal brand review because a translated line is not an exact reproduction of the approved English wording.",
            "The motto does not create a warranty promise or modify the warranty terms. Customer service should answer coverage questions from the written warranty and route a proposed exception to the warranty owner.",
            "Do not add claims such as indestructible, guaranteed for life, or maintenance-free unless approved evidence supports them. The approved motto alone does not substantiate those additional promises.",
            "For a short banner, use the complete motto rather than an ellipsis that could change its meaning. Surrounding copy should remain accurate and should not imply that the slogan changes eligibility or service terms.",
            "The approved line can be used in campaign material, but usage does not authorize alteration of the product name, logo, or visual identity. Follow the current brand asset guide for those elements.",
        ],
        "warranty": [
            "The written warranty covers manufacturing defects for two years from delivery. It excludes accidental damage and normal wear. Eligibility depends on the defect and the product record, so staff should not describe the warranty as covering every failure during the period.",
            "A customer opening a claim should provide the order number and photographs. These materials help the warranty team identify the product and assess the reported issue. If a photograph cannot show the defect, the team may request additional information.",
            "An assessment normally takes five business days after the claim contains enough information to review. This is an assessment estimate, not a promise that a repair or replacement will be completed within five days. Public holidays and missing evidence can affect handling.",
            "The warranty period begins on delivery, not on the date a claim is filed. The team should verify delivery and purchase records before deciding whether a claim is in scope. A late report does not automatically establish or defeat coverage.",
            "Accidental damage and normal wear are outside the stated coverage. If the report mixes a possible manufacturing defect with accidental damage, document both and request a warranty assessment rather than promising an outcome in advance.",
            "The written warranty is the controlling description of product cover. Marketing language, including the approved customer motto, does not expand its terms. Any exception needs explicit authorization from the warranty owner.",
            "Keep claim communications factual: acknowledge receipt, identify missing evidence, and explain the next assessment step. Do not tell a customer that a claim is accepted until the warranty review records that decision.",
            "Order number and photographs are initial evidence, not necessarily the only records needed for a complex case. The warranty team may check product configuration, delivery date, or prior service history before reaching a conclusion.",
        ],
        "support": [
            "Standard cases are acknowledged within one business day, Monday through Friday. Acknowledgement means that support has received the case; it is not a resolution deadline. Requests arriving outside the working week are handled in the next applicable support period.",
            "For an urgent delivery issue, include the order number and dispatch email. The order number locates the purchase and the dispatch message identifies the shipment. Missing either item may require a follow-up before the case can be investigated.",
            "Support can relay the latest carrier estimate but cannot promise a date beyond it. If carrier tracking changes, the updated estimate should be shared with the customer. Avoid converting a current scan into a guaranteed arrival commitment.",
            "Urgency should be explained with the practical impact and relevant deadline. This helps triage, but does not change the published acknowledgement target or authorize support to override carrier operations.",
            "A customer with a tracking link should check the most recent carrier scan while support investigates. The scan is time-sensitive evidence and may change after the case is opened. Record the timestamp when quoting it.",
            "Support should avoid requesting payment-card details or unnecessary personal data in a delivery case. The order reference and dispatch information are the appropriate initial identifiers for locating the shipment.",
            "If a parcel is marked delivered but not received, record the address confirmation and carrier status, then follow the missing-parcel procedure. Do not state that the parcel is lost until the carrier review supports that conclusion.",
            "An acknowledgement can set expectations for the next update without promising a resolution date. If the case requires a carrier response, explain that dependency and keep the customer informed when new evidence arrives.",
        ],
    }
    casework = {
        "returns": [
            "In a first-contact conversation, the adviser should separate three questions: whether the item is eligible, what evidence is needed, and how the customer should send it. Answering only the shipping question can leave the customer thinking eligibility has already been accepted. A useful note records the delivery date, condition reported by the customer, proof reference, and policy version checked, while avoiding a promise about the refund timeline.",
            "For a follow-up case, compare the customer's description with the recorded order and the version effective for that request. If the customer quotes an older answer, acknowledge that the earlier wording may have reflected a prior rule, then explain the current rule and identify what changed. Keep unchanged conditions visible so a revised day limit is not mistaken for a waiver of the other requirements.",
            "A returns handoff should preserve the facts another adviser needs: product, order reference, delivery date, condition, clearance status, and the source version consulted. Do not copy a guess about eligibility into the case as a confirmed decision. If one item in a multi-item order is clearance stock, assess each line separately against its own sale record rather than generalizing from the whole order.",
            "When drafting a public help article, use a concrete example only if it does not create a new exception. State how the delivery date anchors the time period and list the conditions without implying that packaging alone makes an item returnable. A reviewer should check that the article matches the current source and that examples cannot be mistaken for an exhaustive list of customer rights.",
        ],
        "delivery": [
            "An adviser handling a delivery question should first distinguish the general service estimate from a live shipment update. The published duration helps with planning before dispatch, while the carrier record is more relevant after the parcel has left the warehouse. Record which one was used in the response so a later adviser does not repeat a stale estimate as though it were the latest scan.",
            "For a destination check, identify whether the address is within the service area before comparing products. If the destination is not mainland Netherlands, the express duration does not apply even when the customer is willing to pay for speed. Explain the available standard estimate and request a current fulfillment check if stock status or dispatch timing is uncertain.",
            "A delivery comparison should use the same order conditions for both services: in-stock goods, the period after dispatch, and the exclusion of public holidays. If one option is described from order placement and the other from dispatch, the comparison is misleading. Keep the shared assumptions together with both durations and avoid converting a difference into an exact arrival date.",
            "When a target date is important, explain what is known and what remains outside the published estimate. Warehouse readiness, the dispatch event, and the carrier's later scans are distinct stages. A customer can use tracking after dispatch, but neither a service name nor a shorter usual duration authorizes staff to guarantee delivery by a chosen date.",
        ],
        "pricing": [
            "A seat-count estimate should be reproducible from the published components. Record the requested count, allowance, number of added seats, rate per added seat, and base amount. If the requested date falls mid-term or the customer asks about proration, flag that question for billing terms because the simple list-price calculation does not determine how a contract change is charged.",
            "When reviewing an invoice, match its billing period and plan name to the order form before comparing amounts. A correct public monthly rate may not explain a custom contract, tax, or adjustment. Ask finance to reconcile differences using the invoice and signed order form rather than asserting that the list price overrides an agreed commercial document.",
            "A procurement summary should avoid mixing recurring charges with one-time costs. The standard plan facts describe the monthly subscription, included capacity, and additional-seat rate; they do not establish fees for unrelated integrations or bespoke work. Label each amount by period and scope so a reader can compare like with like.",
            "If a team expects to grow, give the customer a formula as well as the current estimate: base monthly fee plus the number of seats beyond the allowance multiplied by the marginal rate. This makes the assumption visible and helps detect whether a later quote includes a different seat count or a negotiated term.",
        ],
        "hosting": [
            "A privacy questionnaire should be answered field by field. Data residency, backup encryption, backup retention, employee communications, and support-ticket processing are distinct categories. If a question combines them, split the answer and state which part the handbook supports; do not let a correct response about production records imply an answer for systems outside that scope.",
            "For a customer asking about a particular attachment, determine where the attachment was submitted and which system retains it before applying the production-record statement. A document sent to support may be part of a ticket rather than the production customer database. The system owner should provide evidence for that separate processing path.",
            "A security response can state that backups are encrypted and retained for thirty days, but should not invent an encryption algorithm, key rotation schedule, deletion guarantee, or disaster-recovery region. Those details need their own approved source. Keep the response narrow enough that the stated control can be checked against the referenced handbook revision.",
            "In a vendor review, retain the exact data category, location, and exclusions beside the answer. This helps legal and security reviewers see whether the statement applies to the requested system. If the requester asks about all data, clarify the categories rather than silently broadening a statement about production customer records.",
        ],
        "people": [
            "A manager reviewing a request should check the employee's available balance and team coverage in the appropriate systems. The handbook gives the general entitlement and approval sequence, but does not reveal any individual's balance or promise approval for particular dates. Keep the decision and the general policy as separate records.",
            "For a part-time schedule, use contracted hours when applying the proration rule and confirm the resulting balance in the leave system. A change in contracted hours during the year may require a recalculation. Do not use a colleague's entitlement or a rough day count as a substitute for the employee's own record.",
            "A team handover should identify coverage for scheduled leave without exposing personal information beyond what teammates need to plan. The leave process concerns both the employee's entitlement and service continuity, but operational planning does not alter the number of days or the requirement for manager approval before booking.",
            "When a question concerns carryover, payout, or a local legal exception, refer it to People Operations with the employee's location and relevant dates. The general allowance does not answer those jurisdiction-specific questions. Avoid turning a broad handbook sentence into a definitive individual employment determination.",
        ],
        "brand": [
            "A copywriter should verify the approved line against the source before final export, including punctuation and capitalization. Drafting from memory can introduce a small wording change that makes the line no longer exact. Keep the motto separate from adjacent claims so a reviewer can check each assertion against its own approved evidence.",
            "If a campaign describes product life or performance, ask the product owner for the written specification and the warranty owner for coverage language. The motto is brand guidance and does not establish durability, repairability, or replacement terms. Review should remove any implication that the slogan expands those documented terms.",
            "A localized campaign should route translated wording through brand review and preserve the approved English line when an exact quote is requested. A translation can be useful for local audiences, but it cannot be described as the exact approved wording unless the localized version has itself been approved.",
            "Before publication, inspect headlines, captions, and landing-page copy together. A compliant motto can still sit beside an unsupported guarantee that changes the overall impression. Review the whole customer-facing passage and make sure every product promise is separately substantiated.",
        ],
        "warranty": [
            "The intake reviewer should capture the order number, delivery date, product identifier, photographs, and the customer's description of the defect. These details help distinguish a potential manufacturing issue from accidental damage or ordinary wear. Intake is not the final coverage decision, so the response should explain that an assessment remains pending.",
            "A status update should distinguish time spent waiting for evidence from the assessment period. The normal five-business-day assessment timing is useful once the claim is ready for review; it does not promise a repair or replacement by that date. If more information is needed, list the missing items and explain that the review can proceed when they arrive.",
            "When a customer reports both a defect and an accident, preserve both parts of the account for the warranty team. Do not discard a possible covered defect solely because the customer also reports damage, and do not tell the customer that the whole issue is covered. The written review determines whether the reported manufacturing defect meets the terms.",
            "A final decision should reference the applicable written warranty and explain the evidence considered. If the product is outside the two-year period, verify the delivery record before communicating that conclusion. Escalate ambiguous dates or product versions rather than using the claim submission date as a substitute for delivery.",
        ],
        "support": [
            "A complete delivery case gives support enough context to locate the right order and shipment without collecting unrelated personal details. The order number and dispatch email are the initial evidence. Add the latest carrier scan and its timestamp when available, then state whether the next step depends on a carrier investigation.",
            "Triage notes should distinguish urgency from evidence. A time-sensitive event can help support prioritize a case, but it does not change the carrier's estimate or permit a delivery promise. State the customer's practical deadline, the current shipment status, and the next expected update as separate facts.",
            "If tracking has not updated, ask when the customer last checked and compare that with the dispatch record. Do not infer that the parcel is lost from a quiet tracking page alone. Follow the missing-parcel procedure and preserve the carrier response so the next adviser can continue the investigation without restarting it.",
            "A good acknowledgement confirms receipt, identifies any missing order or dispatch information, and sets a realistic expectation for the next contact. It should not imply the issue is resolved or promise a date that the carrier has not supplied. If the case is still open after the initial response, send an update when new evidence changes the status.",
        ],
    }
    addenda = {
        "returns": [
            "Example case A concerns a regular item delivered on 4 May, with the customer writing on 20 May and reporting that the item is unused. The adviser records the delivery date, condition, order confirmation, packaging and sale classification before applying the window. The customer is told which facts are established and which still need evidence. The case note does not promise a refund date, because that depends on the later processing steps.",
            "Example case B concerns a clearance item that remains unused and is still in its packaging. The return checklist is not a balancing score in which several conditions can offset one exclusion. The sale classification controls this example, so the adviser explains the exclusion and points the customer to the written terms. If the order record suggests a different classification, the case is held for verification rather than resolved from a screenshot of a later price.",
            "Example case C concerns a customer who has proof of purchase but cannot locate the original package. The adviser records the missing packaging information and checks the applicable written rule before giving a conclusion. Proof establishes the purchase but does not automatically satisfy every other condition. The case is routed for review if the item is oversized or the normal return instructions do not safely cover its transport.",
            "For a multi-item order, prepare one line per product with its sale category, delivery date, condition, evidence and decision. This prevents one clearance item from changing the assessment for another regular item. If a parcel contains products from different deliveries, use the delivery record for each product rather than the earliest date on the combined order.",
            "A policy change review compares the old and new passages side by side, identifies the effective date, and lists which eligibility conditions persist. The communication should not imply retroactive application unless the source says so. If the customer asks about a request made before the effective date, preserve the request timestamp and have the policy owner decide which version governs.",
            "At the end of a returns interaction, summarize the decision, the evidence relied on, and the next customer action. If the case is still under review, say so directly and give the expected next checkpoint rather than implying acceptance. This makes the answer useful to a later adviser without converting an unresolved case into a precedent.",
        ],
        "delivery": [
            "A planning example starts with an item marked in stock at 10:00 on Monday, but the warehouse dispatches it later that afternoon. The published delivery estimate begins at dispatch, so it cannot be added to the time before the parcel leaves. The adviser records the dispatch timestamp before translating a business-day estimate into a customer expectation.",
            "A second example has a public holiday during the delivery period. The holiday is excluded from the business-day estimate, and the exact calendar date still depends on dispatch and carrier activity. Explain the counting convention without claiming the parcel will arrive on a precise date. A current carrier update should be identified with its observation time.",
            "For an express inquiry outside mainland Netherlands, first report the service-area restriction, then state the applicable standard option if the order is in stock. Do not calculate an express arrival date for an ineligible address. A customer may ask for an exception, but the guide does not authorize one and support should not create an informal promise.",
            "A split shipment produces more than one dispatch event. Apply the estimate to each parcel from its own dispatch time and track them independently. The order-level status may remain open while one parcel is already moving. A careful update names the parcel or line item so the customer knows which estimate is being discussed.",
            "When a customer presents a deadline, distinguish a target from an available guarantee. Check stock, destination, service selection and dispatch status. If any of these are unknown, say which fact must be confirmed. A shorter usual duration can inform a choice, but it does not establish that the carrier will meet a date requested by the customer.",
            "The delivery record should preserve the service selected, destination eligibility, inventory status, dispatch timestamp and latest carrier event. These fields support later comparisons and prevent an earlier estimate from being mistaken for a live shipment commitment. Keep the original carrier wording available when an exception or delay is escalated.",
        ],
        "pricing": [
            "A small-team estimate begins with the included seat allowance and counts only the seats above it. A second reviewer should be able to reproduce the total from the base fee, allowance, requested seat count and marginal rate. The quote also states the billing period and plan version so a monthly figure cannot be mistaken for a one-time fee.",
            "A customer increasing seats during an annual term may ask whether the added seats are charged immediately or at renewal. The published monthly rate alone does not answer timing or proration. The adviser records the requested effective date and sends the signed order form to billing for a contract-specific answer.",
            "A procurement sheet often combines subscription, setup, taxes and optional services. Place each item in its own row with amount, frequency and source. The handbook's standard-plan facts should not be used to fill blank values for taxes or custom work. Mark those figures as pending until finance supplies a current quote.",
            "When comparing two team sizes, show the incremental arithmetic separately from the base plan. This makes a change in the marginal rate or included allowance easy to detect. If the customer supplies a negotiated price, retain that offer as separate evidence and do not silently blend it with public list pricing.",
            "Annual invoicing at the start of term is a timing statement. It is not evidence of an annual discount, refund right or cancellation schedule. A response to one of those questions should quote the applicable signed terms or route the issue to the billing owner rather than extrapolate from the monthly list price.",
            "Before sending a final cost summary, reconcile seat count, plan, version, currency, tax treatment and invoice period. If one element is unavailable, label the result an estimate and state the assumption. This lets a customer use the calculation while preserving the distinction between handbook pricing and a binding quote.",
        ],
        "hosting": [
            "A data inventory example separates account records in production from a support message that contains an attachment. The former is in the handbook's stated residency scope; the latter needs its own system record. The reviewer records the category, system owner, location source and date checked so the answer does not overstate what the guide establishes.",
            "A backup question should be decomposed into encryption, retention, restoration and deletion. The source covers encryption and a thirty-day retention period, while the other questions require additional technical evidence. A reviewer leaves those fields open instead of inferring them from the existence of backups.",
            "A customer requests a statement that all data stays in one country. The adviser narrows the answer to production customer records and names the excluded categories. If the customer needs a broader contractual commitment, legal and security review the relevant processor and system inventory before any assurance is sent.",
            "A deletion request and a backup-retention statement can both be accurate without being contradictory: they concern different operations and timelines. The support response should explain the published backup retention carefully, then route deletion timing to the approved privacy process. Do not promise immediate removal from every recovery copy based only on this handbook.",
            "A questionnaire response should preserve evidence provenance. Store the source version and the exact question beside the answer. If an auditor asks about a newly added service, verify whether it handles production customer records, employee email or support tickets before reusing the previous response.",
            "A system map should identify primary storage, replicas, backups, support tools and employee systems separately. The handbook supplies only a portion of that map. Mark unknown locations as requiring confirmation, and avoid turning a single regional statement into a claim about every copy or processor.",
        ],
        "people": [
            "A manager planning coverage for a team records requested dates, work that needs a backup owner, customer handoffs and the approval status. Coverage planning is useful before a decision, but it does not substitute for the approval itself. The employee should not treat a draft schedule as confirmation that leave is booked.",
            "An employee changing contracted hours during the calendar year may need an updated prorated balance. The People team uses the effective date and contract record to calculate the individual amount. A general handbook answer can explain the basis, but should not publish a specific balance without checking the employee's record.",
            "A year-end question about unused leave depends on local law and applicable employment terms. The general entitlement is not enough to establish carryover or payment. The case should include the employee's location, leave year and current balance so People Operations can apply the relevant rule.",
            "For a new starter, the manager should explain how to submit a request, where approval appears, and whom to contact when a decision is delayed. This process guidance should not imply that an entitlement is available before the employee's individual balance has been checked.",
            "A team calendar can reveal coverage conflicts without displaying personal reasons for absence. Share only the information coworkers need to plan. Keep medical or other sensitive explanations in the approved People channel rather than the general project conversation.",
            "A leave handoff records customer commitments, delegated decisions and return-to-work contacts. It should not include a guess about the employee's entitlement or approval. When schedule pressure exists, the manager discusses options with the employee and records the final decision through the formal process.",
        ],
        "brand": [
            "A campaign review examines the headline, caption, product page and paid advertisement as one customer journey. A safe slogan in the headline can be undermined by an unsupported guarantee in a nearby caption. Reviewers list each factual product claim, identify its evidence and remove or qualify claims that lack an approved source.",
            "A localization example keeps the approved English motto available while the translated campaign is under review. The translation is checked for meaning, legal implications and local brand rules. Until approved, it should not be labeled an exact quotation of the English line.",
            "A retailer asks to add a lifetime coverage badge beside the company motto. The brand reviewer separates the motto from the warranty claim and sends the badge for warranty approval. If no written warranty supports the badge, the reviewer declines it even if the surrounding campaign uses the approved wording.",
            "A social post may shorten surrounding copy, but the approved motto should remain intact when presented as the official line. Ellipses, word substitutions or punctuation changes make a paraphrase, not an exact quote. Save the approved asset and final rendered preview together for review.",
            "A product demonstration can make an implied claim through imagery as well as text. Review the entire frame for suggestions of indestructibility, unlimited use or guaranteed performance. The motto itself does not substantiate those visual implications; product evidence and warranty language must be reviewed separately.",
            "A correction after publication should identify the exact asset, affected channels, current wording and approval owner. Replace the inaccurate claim promptly and retain the corrected version for audit. Do not merely remove the motto if the actual problem is an adjacent unsupported promise.",
        ],
        "warranty": [
            "A claim triage record links order, delivery date, product configuration, photographs, defect description and any prior service. The initial reviewer checks that the reported issue is plausibly a manufacturing defect, while leaving the final coverage decision to the warranty assessment. Missing information is requested specifically rather than using a blanket refusal.",
            "A mixed-cause report describes both a manufacturing issue and accidental impact. Preserve the customer's timeline and images for the assessor; do not classify the whole event from one sentence. The written warranty excludes accidental damage but the evidence may still help the reviewer distinguish separate defects.",
            "A five-business-day assessment estimate begins only when the claim is ready for assessment. An incomplete submission may need a request for evidence first. The customer update should explain the current stage, missing items if any, and the difference between assessment timing and repair completion.",
            "A product arriving near the end of the warranty period should be assessed from the delivery record, not the date the complaint was typed. If a delivery record is disputed, request the carrier confirmation and pause the conclusion. A timely report does not itself establish that the defect is covered.",
            "The warranty team documents the evidence considered, the applicable written term, and the reason for its decision. A customer-facing response can summarize that reasoning without exposing internal comments. If the written term is ambiguous, escalate to the owner instead of inventing an exception.",
            "A repeated failure after repair should include the earlier case reference and repair outcome. That history may help determine the next step, but it does not replace the current defect assessment. Keep each claim tied to the product record so a similar issue on another item is not conflated.",
        ],
        "support": [
            "An urgent case record contains the customer deadline, order number, dispatch message, parcel identifier, destination and latest carrier scan. Support should state which of these facts has been verified. Urgency informs triage and update frequency; it does not allow the adviser to promise a carrier outcome that has not been confirmed.",
            "When tracking is unchanged for several days, compare the carrier event history with dispatch time and service estimate. Open the missing-parcel process when its criteria are met and keep the carrier case reference. Do not tell the customer a parcel is lost merely because a tracking page has not refreshed.",
            "An acknowledgement should confirm receipt and name any next action, such as supplying the order number or waiting for a carrier response. It should not imply a full investigation is complete. If the next update depends on a third party, say that clearly and provide a realistic checkpoint rather than a guaranteed resolution date.",
            "A delivered-but-not-received report requires checking the address, delivery scan and any available proof of delivery. Record the time and source of each item. A scan is evidence of a carrier event, not conclusive proof that the intended customer personally received the parcel.",
            "If a customer replies in a new email thread, attach the message to the existing case using its reference and verify the order before sharing details. Avoid asking for payment credentials or unrelated identity documents. The case should keep one chronology of customer statements, carrier updates and support actions.",
            "A final support response distinguishes what happened, what the carrier confirmed, and what the company can do next. If evidence remains incomplete, leave the case status open and explain the follow-up. Do not close a case just because an acknowledgement was sent within the standard response target.",
        ],
    }
    topic = (
        "returns"
        if key == "returns_old"
        else "delivery"
        if key in ("standard", "express")
        else "support"
        if key == "operations"
        else key
    )
    prose = "\n\n".join(
        f"{company} handbook guidance: {p}"
        for p in [*sections[topic], *casework[topic], *addenda[topic]]
    )
    c = next(v for v in COMPANIES.values() if v["name"] == company)
    return prose + "\n\n" + case_appendix(key, company, thread_num, c)


def case_appendix(key: str, company: str, thread_num: int, c: dict) -> str:
    """Seeded, distinct customer/business records make each handbook section queryable."""
    rows = ["| Case | Record | Evidence / decision fields |", "|---|---|---|"]
    prefix = {
        "returns": "RET",
        "returns_old": "RET",
        "delivery": "SHP",
        "standard": "SHP",
        "express": "SHP",
        "pricing": "BIL",
        "hosting": "DAT",
        "people": "PPL",
        "brand": "CMP",
        "warranty": "WRT",
        "support": "SUP",
        "operations": "ONB",
    }[key]
    for n in range(1, 61):
        case = f"{prefix}-{thread_num + 1:02d}-{n:03d}"
        a = (n * 3 + thread_num) % 29
        b = (n * 7 + thread_num) % 23
        if prefix == "RET":
            state = "unused" if n % 3 else "opened"
            proof = "order-confirmation" if n % 4 else "receipt-pending"
            clearance = "regular" if n % 5 else "clearance"
            detail = f"delivered 2026-04-{a + 1:02d}; inquiry 2026-05-{b + 1:02d}; {state}; {proof}; {clearance}; assess against effective window and record each condition"
        elif prefix == "SHP":
            service = "express" if n % 2 else "standard"
            days = c["express"] if service == "express" else c["standard"]
            dest = "mainland NL" if n % 4 else "outside express area"
            detail = f"in-stock; {service}; {days} business days after dispatch; {dest}; holiday excluded; tracking event {a + 1} May; estimate is not a delivery guarantee"
        elif prefix == "BIL":
            seats = 25 + (n * 5 + thread_num) % 26
            total = c["budget"] + max(0, seats - 25) * 18
            detail = f"team plan; {seats} seats; €{c['budget']} base includes 25; €18 per extra seat; list monthly total €{total}; annual invoice at term start"
        elif prefix == "DAT":
            category = "production customer records" if n % 3 else "support ticket"
            detail = f"request concerns {category}; production location {c['region']}, Netherlands; backup encrypted; backup retention 30 days; confirm scope before replying"
        elif prefix == "PPL":
            hours = "full-time" if n % 2 else "part-time"
            entitlement = (
                f"{c['leave']} annual paid days"
                if hours == "full-time"
                else "prorated by contracted hours"
            )
            detail = f"employee schedule {hours}; entitlement {entitlement}; manager approval required before booking; individual balance must be checked in People system"
        elif prefix == "CMP":
            detail = f"asset batch {thread_num + 1:02d}-{n:03d}; approved line ‘Made for everyday journeys.’; exact punctuation; warranty remains written limited cover; legal sign-off recorded separately"
        elif prefix == "WRT":
            issue = "manufacturing defect" if n % 3 else "accidental damage"
            detail = f"delivered 2026-04-{a + 1:02d}; reported 2026-05-{b + 1:02d}; issue {issue}; two-year written term; order number and photographs requested; assess before outcome"
        elif prefix == "SUP":
            detail = f"urgent parcel case; order and dispatch email recorded; acknowledgement target one business day Mon-Fri; carrier estimate {b + 1} days; next update after carrier scan"
        else:
            detail = f"onboarding checkpoint {n:03d}; verify source version, company scope, customer question and owner; unresolved items assigned; no unsupported guarantee; next review {a + 1} May"
        rows.append(f"| {case} | {detail} |")
    return "## Seeded case register\n\n" + "\n".join(rows)


def question(cid: str, index: int, case_index: int = 0) -> tuple[str, list[str], str]:
    c = COMPANIES[cid]
    n = c["name"]
    plan = [
        (
            f"For the customer onboarding note, what return rules should I state for {n}?",
            ["returns"],
            "List the current return window and the continuing proof-of-purchase, packaging, and clearance conditions.",
        ),
        (
            f"What delivery estimate should the note give for in-stock standard and express orders at {n}?",
            ["delivery"],
            "Compare both in-stock estimates, state they begin after dispatch, and retain the mainland limitation for express.",
        ),
        (
            f"How should I describe the {n} team plan price and included seats?",
            ["pricing"],
            "State the monthly price, included seat count, and marginal extra-seat price.",
        ),
        (
            f"Where are {n}'s EU production customer records stored, and what is outside that statement?",
            ["hosting"],
            "Give the storage location and distinguish production customer records from the excluded categories.",
        ),
        (
            f"What annual leave and approval rule applies to a full-time {n} employee?",
            ["people"],
            "State annual paid leave and approval before booking.",
        ),
        (
            f"Quote {n}'s approved customer motto exactly and note the warranty wording restriction.",
            ["brand"],
            "Quote the motto verbatim and avoid implying a lifetime guarantee.",
        ),
        (
            f"What does the {n} written warranty cover, and how does a customer start a claim?",
            ["warranty"],
            "State duration, covered defect type, key exclusions, and claim materials.",
        ),
        (
            f"What response time and information should a customer expect for an urgent {n} delivery issue?",
            ["support"],
            "State acknowledgement timing, business-day scope, and the information to provide.",
        ),
        (
            "Return to the earlier returns guidance: which window applies now, and which conditions stayed the same?",
            ["returns"],
            "Use the current revision for the window and preserve its continuing conditions.",
        ),
        (
            f"For a mainland customer with an in-stock order, how much faster is {n} express than standard?",
            ["delivery"],
            "Calculate the difference from both stated estimates and retain their shared scope.",
        ),
        (
            f"If the team needs 28 seats on the {n} plan, what is the monthly total?",
            ["pricing"],
            "Calculate the base fee plus three additional seats at the stated per-seat rate.",
        ),
        (
            f"Can the onboarding note say employee email is stored in {c['region']}? Explain the scope.",
            ["hosting"],
            "Do not extend the production customer-record residency claim to employee email.",
        ),
        (
            f"Turn the return guidance into a concise checklist for a customer of {n}.",
            ["returns"],
            "Include current window, unused condition, proof, original packaging, and clearance exclusion.",
        ),
        (
            f"Would {n} standard delivery meet a three-business-day target for in-stock goods?",
            ["delivery"],
            "Compare standard estimate to target without turning an estimate into a guarantee.",
        ),
        (
            f"What should a full-time {n} employee do before booking leave?",
            ["people"],
            "Manager approval must happen before booking.",
        ),
        (
            f"Draft one short {n} campaign line using the exact motto and no unsupported guarantee.",
            ["brand"],
            "Use exact motto and do not imply lifetime coverage.",
        ),
        (
            f"How long does a {n} warranty assessment normally take, and what should be submitted?",
            ["warranty"],
            "State assessment duration and order number plus photographs.",
        ),
        (
            f"What details should accompany an urgent delivery case to {n} support?",
            ["support"],
            "Include the order number and dispatch email; acknowledge response-time scope.",
        ),
        (
            f"Compare {n}'s current return period with its original period. How much changed?",
            ["returns"],
            "Compare current and superseded windows and give the increase in days.",
        ),
        (
            f"Which {n} delivery option fits a speed-sensitive mainland order, and what caveats apply?",
            ["delivery"],
            "Recommend based on the two estimates, with availability and estimate caveats.",
        ),
        (
            f"What would 30 seats cost monthly on the {n} plan?",
            ["pricing"],
            "Calculate five added seats on top of the included 25.",
        ),
        (
            f"Summarize {n}'s backup protection and retention without broadening the residency claim.",
            ["hosting"],
            "State encryption, 30-day retention, and the production-record scope.",
        ),
        (
            f"How is leave allowance calculated for a part-time {n} employee?",
            ["people"],
            "Allowance is prorated by contracted hours.",
        ),
        (
            f"Can campaign copy promise lifetime product coverage for {n}?",
            ["brand"],
            "No; coverage is limited to the written warranty.",
        ),
        (
            f"What evidence should a {n} customer gather for a manufacturing-defect warranty claim?",
            ["warranty"],
            "Provide order number and photographs; claim relates to manufacturing defects.",
        ),
        (
            f"What delivery date may {n} support promise beyond the carrier estimate?",
            ["support"],
            "Support cannot promise a date beyond the current carrier estimate.",
        ),
        (
            f"Please restate the current {n} return rules in two sentences for the onboarding note.",
            ["returns"],
            "State the current period and all material conditions.",
        ),
        (
            f"For a mainland order, summarize both {n} delivery estimates and their shared qualification.",
            ["delivery"],
            "State both durations, dispatch start, and public-holiday exclusion.",
        ),
        (
            f"Give the {n} plan price for 25 seats and explain how annual billing is invoiced.",
            ["pricing"],
            "State monthly base price and annual invoice timing.",
        ),
        (
            f"Using the accumulated onboarding facts, summarize {n}'s customer-facing return and delivery guidance.",
            ["returns", "delivery"],
            "Combine current return conditions with qualified standard/express delivery estimates.",
        ),
    ]
    cycle_index = index % (len(plan) + 6)
    if cycle_index < len(plan):
        q, keys, requirement = plan[cycle_index]
    else:
        extras = [
            (
                f"Which facts should a {n} adviser verify before sending the final onboarding response?",
                ["returns", "delivery"],
                "Confirm current return terms and qualified delivery estimates.",
            ),
            (
                f"Summarize {n}'s monthly team-plan pricing and which figures are separate line items.",
                ["pricing"],
                "Separate base fee, included seats, and incremental seat price.",
            ),
            (
                f"Give a scoped privacy answer about {n}'s record location, encrypted backup, and exclusions.",
                ["hosting"],
                "Include location, backup controls and the explicit scope boundaries.",
            ),
            (
                f"What can {n} promise about a manufacturing-defect claim's coverage and timing?",
                ["warranty"],
                "State coverage and assessment time as qualified terms, not an outcome guarantee.",
            ),
            (
                f"Prepare a support handoff for a late {n} parcel using the documented process.",
                ["support", "delivery"],
                "Use order/dispatch evidence and current carrier estimate without promising a date.",
            ),
            (
                f"Which annual leave details can {n} publish generally, and which require an employee record?",
                ["people"],
                "State general entitlement and proration while noting individual balances need records.",
            ),
        ]
        q, keys, requirement = extras[cycle_index - len(plan)]
    phases = [
        "initial requirements gathering",
        "policy validation",
        "drafting and approval",
        "customer support exercises",
        "final publication audit",
    ]
    phase = phases[(index // 36) % len(phases)]
    q = f"For {phase}, {q[0].lower() + q[1:]}"
    q = q.replace("28 seats", f"{27 + case_index} seats").replace(
        "30 seats", f"{30 + case_index} seats"
    )
    return q, keys, requirement


def make_thread(
    tid: str, cid: str, split: str, thread_num: int
) -> tuple[dict, list[dict]]:
    c = COMPANIES[cid]
    d = docs(cid)
    start = BASE + timedelta(days=thread_num * 2 + (70 if split == "heldout" else 0))
    cursor = start
    thread = {
        "company_id": cid,
        "thread_id": tid,
        "scenario_id": "long_coherent_onboarding",
        "provenance": "synthetic_long_scale_control; source-family-and-cadence-grounded_in_rag_workload; answer_templates_not_observed_model_outputs",
        "runs": [],
    }
    labels = []
    history: list[str] = []
    prior_parts: dict[str, dict] = {}
    design = []
    medium_gap_turn = 34 + thread_num % 5
    long_gap_turn = 60 + thread_num % 7
    for i in range(1, TURN_COUNT + 1):
        q, keys, requirement = question(cid, i - 1, thread_num)
        family = "direct_lookup"
        if i == 1:
            q = f"For the returns section, please keep your answer to the follow-up in one sentence. What was {c['name']}'s original return window and conditions?"
            keys = ["returns_old"]
            requirement = "State the original return window and all listed eligibility conditions."
            family = "constraint_carryover"
        elif i == 6:
            q = f"Quote {c['name']}'s approved customer motto exactly."
            keys = ["brand"]
            requirement = "Quote the approved motto exactly."
            family = "exact_quote"
        elif i == 7:
            q = "Returning to the return policy, restate the current conditions in one sentence."
            keys = ["returns"]
            requirement = (
                "State the current return period and conditions in one sentence."
            )
            family = "constraint_carryover"
        elif i == 9:
            q = f"Compare the original {c['name']} return window with the current one. What changed?"
            keys = ["returns"]
            requirement = "Compare the original and current return periods and identify the effective revision."
            family = "version_supersession"
        elif i == 10:
            q = f"For in-stock orders, compare {c['name']} standard and express delivery and the express service limit."
            keys = ["standard", "express"]
            requirement = (
                "Compare both in-stock durations and state express availability."
            )
            family = "multi_source_comparison"
        elif i == 11:
            case_no = 1 + thread_num % 10
            seats = 25 + (case_no * 5 + thread_num) % 26
            q = f"For pricing case BIL-{thread_num + 1:02d}-{case_no:03d}, what monthly total is recorded?"
            keys = ["pricing"]
            requirement = (
                "Report the seat count and recorded monthly fee from this case."
            )
            family = "structured_case_lookup"
        elif i == 15:
            q = f"Remind me of {c['name']}'s current return window and conditions."
            keys = ["returns"]
            requirement = "State the current return period and conditions."
            family = "fresh_duplicate"
        elif i == 18:
            q = "Turn the approved motto you quoted earlier into a heading, preserving it exactly."
            keys = []
            requirement = "Reproduce the earlier approved motto exactly."
            family = "output_reference"
        elif i == 21:
            case_no = 11 + thread_num % 10
            seats = 25 + (case_no * 5 + thread_num) % 26
            q = f"For pricing case BIL-{thread_num + 1:02d}-{case_no:03d}, verify the monthly total from its seat count."
            keys = ["pricing"]
            requirement = (
                "Verify the seat count and recorded monthly fee in the case record."
            )
            family = "structured_case_lookup"
        elif i == 30:
            q = f"What original return window did the first policy state for {c['name']}?"
            keys = ["pricing"]
            requirement = (
                "Recall the original return window from the first policy source."
            )
            family = "old_retrieval"
        elif i == 36:
            q = "The current return policy was updated earlier in our review. Which window applies now?"
            keys = ["returns"]
            requirement = "State the currently effective return window from the earlier revision comparison."
            family = "unselected_decoy"
        elif i == 37:
            q = "Returning to my first return-policy question, what is the current window?"
            keys = ["returns"]
            requirement = "Give the current return period; the explicit topic reference is resolved by the current source."
            family = "topic_return"
        answer_keys = list(keys)
        related = {
            "returns": ["returns", "warranty", "support", "operations", "people"],
            "returns_old": [
                "returns_old",
                "warranty",
                "support",
                "operations",
                "people",
            ],
            "delivery": ["standard", "express", "support", "operations", "returns"],
            "standard": ["standard", "express", "support", "operations", "returns"],
            "express": ["express", "standard", "support", "operations", "returns"],
            "pricing": ["pricing", "operations", "people", "support", "hosting"],
            "hosting": ["hosting", "operations", "support", "people", "pricing"],
            "people": ["people", "operations", "support", "pricing", "hosting"],
            "brand": ["brand", "operations", "warranty", "support", "returns"],
            "warranty": ["warranty", "returns", "support", "operations", "brand"],
            "support": ["support", "standard", "operations", "warranty", "returns"],
        }
        if i == 18:
            wanted = ["operations", "support", "pricing", "hosting"]
        elif i == 36:
            wanted = ["returns_old", "warranty", "support", "operations"]
        else:
            wanted = list(answer_keys)
            for key in answer_keys:
                for related_key in related.get(key, [key]):
                    if related_key not in wanted:
                        wanted.append(related_key)
            # A multi-source comparison can produce five semantically related passages.
            wanted = wanted[:5]
        rid = f"{tid}-run-{i:03d}"
        input_id = f"{tid}-r{i:03d}-input"
        output_id = f"{tid}-r{i:03d}-output"
        rparts = []
        current_ids = []
        visible_ids = []
        source_objs = []
        source_by_key = {}
        for j, key in enumerate(wanted, 1):
            base_text, document_family, version = d[key]
            text = base_text + "\n\n" + handbook_expansion(key, c["name"], thread_num)
            pid = f"{tid}-r{i:03d}-retrieval{j}"
            exposed = not (i == 36 and key == "returns_old")
            rparts.append(
                {
                    "part_id": pid,
                    "document_id": f"{cid}-{document_family}-{thread_num + 1:02d}",
                    "version": version,
                    "chunk_id": key,
                    "text": text,
                    "exposed_to_model": exposed,
                }
            )
            current_ids.append(pid)
            if exposed:
                visible_ids.append(pid)
            quote = base_text
            if key == "pricing" and i in (11, 21):
                case_no = (1 + thread_num % 10) if i == 11 else (11 + thread_num % 10)
                quote = next(
                    line
                    for line in text.splitlines()
                    if f"BIL-{thread_num + 1:02d}-{case_no:03d}" in line
                )
            src = {"part_id": pid, "quote": quote}
            source_objs.append(src)
            source_by_key[key] = src
        # Templates are scaffolding for native answer generation, not claimed as observed answers.
        if i == 1:
            answer = f"{c['name']} accepts unused goods within {c['base']} days after delivery, with proof of purchase and original packaging; clearance goods are excluded."
        elif i == 6:
            answer = "‘Made for everyday journeys.’"
        elif i == 7:
            answer = f"The current return window is {c['base'] + 7} days after delivery; goods must be unused, proof of purchase is required, clearance goods are excluded, and items must be in original packaging."
        elif i == 9:
            answer = f"The window changed from {c['base']} to {c['base'] + 7} days on 1 March 2026; the other conditions remain in force."
        elif i == 10:
            answer = f"For in-stock orders, standard takes {c['standard']} business days and express takes {c['express']} after dispatch; express is mainland Netherlands only."
        elif i == 11:
            case_no = 1 + thread_num % 10
            seats = 25 + (case_no * 5 + thread_num) % 26
            total = c["budget"] + max(0, seats - 25) * 18
            answer = f"Pricing case BIL-{thread_num + 1:02d}-{case_no:03d} records {seats} seats and a monthly total of €{total}."
        elif i == 15:
            answer = f"The current return window is {c['base'] + 7} days after delivery; proof is required, clearance goods are excluded, and items must be unused and in original packaging."
        elif i == 18:
            answer = "‘Made for everyday journeys.’"
        elif i == 21:
            case_no = 11 + thread_num % 10
            seats = 25 + (case_no * 5 + thread_num) % 26
            total = c["budget"] + max(0, seats - 25) * 18
            answer = f"Pricing case BIL-{thread_num + 1:02d}-{case_no:03d} records {seats} seats and a monthly total of €{total}."
        elif i == 30:
            answer = f"The original policy allowed returns within {c['base']} days after delivery."
        elif i in (36, 37):
            answer = (
                f"The current return window is {c['base'] + 7} days after delivery."
            )
        elif keys:
            answer = " ".join(d[k][0] for k in keys)
        else:
            answer = "I can summarize the documented customer guidance from the sources already discussed."
        run_start = cursor
        gs = run_start
        ge = gs + timedelta(seconds=1)
        end = ge
        run = {
            "run_id": rid,
            "run_index": i,
            "started_at": iso(run_start),
            "ended_at": iso(end),
            "history_policy": "full_canonical",
            "input": {"part_id": input_id, "text": q},
            "output": {"part_id": output_id, "text": answer},
            "retrievals": rparts,
            "generation": {
                "observation_id": digest(rid),
                "started_at": iso(gs),
                "ended_at": iso(ge),
                "model": "gpt-6-luna",
                "usage": None,
                "request_part_ids": history + [input_id] + visible_ids,
            },
        }
        thread["runs"].append(run)
        historical_sources = []
        task_specs = []
        if i == 7:
            task_specs.append(
                (
                    requirement,
                    "retrieval",
                    [{"sources": [source_by_key[k] for k in answer_keys]}],
                    "no",
                )
            )
            source = {
                "part_id": f"{tid}-r001-input",
                "quote": "For the returns section, please keep your answer to the follow-up in one sentence.",
            }
            task_specs.append(
                (
                    "Apply the earlier one-sentence response constraint.",
                    "input",
                    [{"sources": [source]}],
                    "yes",
                )
            )
            historical_sources = [source]
        elif i == 9:
            old = citation(prior_parts[f"{tid}-r001-retrieval1"])
            task_specs.append(
                (
                    requirement,
                    "retrieval",
                    [{"sources": [old, source_by_key["returns"]]}],
                    "yes",
                )
            )
            historical_sources = [old]
        elif i == 15:
            old = citation(prior_parts[f"{tid}-r009-retrieval1"])
            task_specs.append(
                (
                    requirement,
                    "retrieval",
                    [{"sources": [source_by_key["returns"]]}, {"sources": [old]}],
                    "no",
                )
            )
            historical_sources = [old]
        elif i == 18:
            old = citation(prior_parts[f"{tid}-r006-output"])
            task_specs.append((requirement, "output", [{"sources": [old]}], "yes"))
            historical_sources = [old]
        elif i == 30:
            old = citation(prior_parts[f"{tid}-r001-retrieval1"])
            task_specs.append((requirement, "retrieval", [{"sources": [old]}], "yes"))
            historical_sources = [old]
        elif i == 36:
            old = citation(prior_parts[f"{tid}-r009-retrieval1"])
            task_specs.append((requirement, "retrieval", [{"sources": [old]}], "yes"))
            historical_sources = [old]
        else:
            task_specs.append(
                (
                    requirement,
                    "retrieval" if answer_keys else "output",
                    [
                        {
                            "sources": [source_by_key[k] for k in answer_keys]
                            or [citation(prior_parts[f"{tid}-r006-output"])]
                        }
                    ],
                    "no" if answer_keys else "yes",
                )
            )
        reqs = []
        claim_reqs = []
        for j, (desc, kind, alts, needed) in enumerate(task_specs, 1):
            req_id = f"req-{j}"
            reqs.append(
                {
                    "requirement_id": req_id,
                    "description": desc,
                    "source_kind": kind,
                    "necessity": "inferred",
                    "alternatives": alts,
                    "evidence_note": "Authored source requirements for the fixture; not a model judgment.",
                    "question_quote": q,
                    "fulfillment": "fulfilled",
                    "historical_context_needed": needed,
                }
            )
            claim_reqs.append(
                {
                    "requirement_id": req_id,
                    "description": desc,
                    "source_kind": kind,
                    "necessity": "inferred",
                    "alternatives": alts,
                    "evidence_note": "The template answer is grounded in the cited evidence.",
                }
            )
        hist_edges = []
        for src in historical_sources:
            oldpart = prior_parts[src["part_id"]]
            targets = (
                ["req-2"]
                if i == 7 and oldpart["kind"] == "input"
                else [x["requirement_id"] for x in reqs]
            )
            hist_edges.append(
                {
                    "source_part_id": src["part_id"],
                    "source_run_id": oldpart["origin_run_id"],
                    "source_kind": oldpart["kind"],
                    "relation": "possibly_used",
                    "target_claim_ids": ["claim-1"],
                    "target_task_requirement_ids": targets,
                    "evidence_note": "An earlier turn supplies information or a conversation instruction required by this request.",
                }
            )
        claim = {
            "claim_id": "claim-1",
            "target_quote": answer,
            "relation": "supported",
            "requirements": claim_reqs,
        }
        labels.append(
            {
                "schema_version": "compache-dependency-label-v1",
                "thread_id": tid,
                "target_run_id": rid,
                "status": "complete",
                "coverage_notes": [
                    "Authored expected source requirements for this fixture; not a model judgment."
                ],
                "task_requirements": reqs,
                "claims": [claim],
                "historical_dependencies": hist_edges,
                "unresolved": [],
            }
        )
        design.append(
            {
                "thread_id": tid,
                "run_id": rid,
                "run_index": i,
                "question_family": family,
                "historical_dependency_lags": [
                    i - prior_parts[s["part_id"]]["origin_run_index"]
                    for s in historical_sources
                ],
                "selected_retrieval_count": len(visible_ids),
                "retrieval_candidate_count": len(rparts),
                "selected_chunk_refs": [
                    {
                        "document_id": p["document_id"],
                        "version": p["version"],
                        "chunk_id": p["chunk_id"],
                        "exposed_to_model": p["exposed_to_model"],
                    }
                    for p in rparts
                ],
            }
        )
        for item in [run["input"], run["output"], *rparts]:
            prior_parts[item["part_id"]] = {
                "part_id": item["part_id"],
                "origin_run_id": rid,
                "origin_run_index": i,
                "kind": "input"
                if item["part_id"] == input_id
                else "output"
                if item["part_id"] == output_id
                else "retrieval",
                "text": item["text"],
            }
        history += [input_id] + visible_ids + [output_id]
        gap = (
            2400
            if i == long_gap_turn
            else 420
            if i == medium_gap_turn
            else 2 + ((i + thread_num) % 2)
        )
        cursor = end + timedelta(seconds=gap)
    return thread, labels, design


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows),
        encoding="utf-8",
    )


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    threads = []
    labels = []
    corpus = []
    design = []
    split_map = {}
    for num, (tid, cid, split) in enumerate(THREADS):
        t, ls, ds = make_thread(tid, cid, split, num)
        threads.append(t)
        labels.extend(ls)
        design.extend(ds)
        split_map[tid] = split
    for num, (tid, cid, split) in enumerate(THREADS):
        for key, (base_text, family, version) in docs(cid).items():
            text = (
                base_text
                + "\n\n"
                + handbook_expansion(key, COMPANIES[cid]["name"], num)
            )
            corpus.append(
                {
                    "company_id": cid,
                    "thread_id": tid,
                    "document_id": f"{cid}-{family}-{num + 1:02d}",
                    "version": version,
                    "chunk_id": key,
                    "text": text,
                }
            )
    write_jsonl(OUT / "threads.jsonl", threads)
    write_jsonl(OUT / "labels.jsonl", labels)
    write_jsonl(OUT / "corpus.jsonl", corpus)
    write_jsonl(OUT / "question_design.jsonl", design)
    manual = [
        "# Fictional Company Onboarding Handbook",
        "",
        "This is a synthetic case manual for long-context RAG evaluation. Each section has a stable document/version/chunk reference; seeded registers contain distinct records for each conversation. All companies, customer cases, and transaction IDs are fictional.",
        "",
    ]
    for row in corpus:
        manual += [
            f"## {row['company_id'].title()} / {row['document_id']}",
            "",
            f"Source reference: `{row['document_id']}@{row['version']}#{row['chunk_id']}`",
            "",
            row["text"],
            "",
        ]
    (OUT / "handbook.md").write_text("\n".join(manual), encoding="utf-8")
    words_by_thread = {
        t["thread_id"]: sum(
            len(
                " ".join(
                    (p.get("text") or "")
                    for p in r["retrievals"]
                    if p["exposed_to_model"]
                ).split()
            )
            for r in t["runs"]
        )
        for t in threads
    }
    hashes = {
        name: hashlib.sha256((OUT / name).read_bytes()).hexdigest()
        for name in (
            "threads.jsonl",
            "labels.jsonl",
            "corpus.jsonl",
            "question_design.jsonl",
            "handbook.md",
        )
    }
    family_counts = {}
    for row in design:
        family_counts[row["question_family"]] = (
            family_counts.get(row["question_family"], 0) + 1
        )
    final_prompt_words = {
        t["thread_id"]: sum(
            len((p.get("text") or "").split())
            for r in t["runs"]
            for p in [
                r["input"],
                *[x for x in r["retrievals"] if x["exposed_to_model"]],
            ]
        )
        + sum(len((r["output"].get("text") or "").split()) for r in t["runs"][:-1])
        for t in threads
    }
    medium_after = {tid: 34 + num % 5 for num, (tid, cid, split) in enumerate(THREADS)}
    long_after = {tid: 60 + num % 7 for num, (tid, cid, split) in enumerate(THREADS)}
    manifest = {
        "schema_version": "compache-rag-long-fixture-v1",
        "seed": SEED,
        "model": "gpt-6-luna",
        "train_thread_ids": [k for k, v in split_map.items() if v == "train"],
        "heldout_thread_ids": [k for k, v in split_map.items() if v == "heldout"],
        "frozen_split_before_outcomes": True,
        "controlled_length_scale": {
            "source_distribution": "rag_workload question-family/source dependency patterns and three company profiles",
            "source_median_turns": 5,
            "long_turns_per_thread": TURN_COUNT,
            "rationale": "A deliberate length-scale extension in both turn count and retrieval passage size. The full_canonical trace retains source dependency families and accumulates detailed company handbook content; no synthetic token padding. This separate long-scale training distribution is necessary to model full native-context behavior; short training fixtures do not cover long prompts.",
        },
        "thread_count": len(threads),
        "turn_count": sum(len(t["runs"]) for t in threads),
        "turns_per_thread": TURN_COUNT,
        "retrieval_chunk_chars": {
            "min": min(len(x["text"]) for x in corpus),
            "max": max(len(x["text"]) for x in corpus),
        },
        "retrieval_word_counts_by_thread": words_by_thread,
        "retrieval_word_counts_per_turn": {
            "min": min(
                len(p["text"].split())
                for t in threads
                for r in t["runs"]
                for p in r["retrievals"]
                if p["exposed_to_model"]
            ),
            "max": max(
                len(p["text"].split())
                for t in threads
                for r in t["runs"]
                for p in r["retrievals"]
                if p["exposed_to_model"]
            ),
        },
        "final_full_history_prompt_words": final_prompt_words,
        "final_prompt_token_estimate_range": {
            "method": "word-count planning estimate only; replace with native model token counts",
            "low": int(min(final_prompt_words.values()) * 1.25),
            "high": int(max(final_prompt_words.values()) * 1.5),
        },
        "per_run_retrieval_word_counts": {
            "min": min(
                sum(
                    len(p["text"].split())
                    for p in r["retrievals"]
                    if p["exposed_to_model"]
                )
                for t in threads
                for r in t["runs"]
            ),
            "max": max(
                sum(
                    len(p["text"].split())
                    for p in r["retrievals"]
                    if p["exposed_to_model"]
                )
                for t in threads
                for r in t["runs"]
            ),
        },
        "cadence": {
            "warm_inter_run_gap_seconds": [2, 3],
            "offline_medium_return_gap_seconds": 420,
            "offline_long_return_gap_seconds": 2400,
            "medium_gap_after_turn_by_thread": medium_after,
            "long_gap_after_turn_by_thread": long_after,
            "note": "synthetic trace timestamps for offline retention scenarios only; run all native calls immediately, then replay measured timings in warm/medium/long scenario comparisons",
        },
        "question_family_counts": family_counts,
        "historical_dependency_lag_counts": {
            str(lag): sum(lag in row["historical_dependency_lags"] for row in design)
            for lag in sorted(
                {lag for row in design for lag in row["historical_dependency_lags"]}
            )
        },
        "usage": "null; tokens must be measured by native run harness; word counts do not substitute for provider usage",
        "answer_status": "output fields are deterministic scaffolding/templates and MUST be replaced by actual native model answers in OpenCode and Compache runs",
        "sha256": hashes,
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "threads": len(threads),
                "turns": manifest["turn_count"],
                "hashes": hashes,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    OUT = parser.parse_args().output
    build()
