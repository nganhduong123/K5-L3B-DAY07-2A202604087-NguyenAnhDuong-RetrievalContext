"""Advanced benchmark (dev/test) for OpenStax "Business Ethics" — written from the ORIGINAL pages/crops,
before any retrieval or agent run, and without looking at model-generated figure descriptions.

Run: python -m src.advanced.benchmark   -> data/advanced/benchmark.json (+ validation)

Each answerable item: group (table | image | chart | multihop | text), question, expected answer,
evidence = list of REQUIRED pieces {unit, page, kind, locator}; all pieces must be found/cited
(multi-hop needs every piece). `check` strings must literally occur in the parsed unit text
(tables/text) — figures are checked against the crop by eye (values noted in `locator`).
Unanswerable items: `absence_check` keywords searched over every unit (must return no hit except
listed `allowed_hits`), plus a written reason.
"""
import json
import re

from ..config import ROOT

UNITS = ROOT / "data" / "advanced" / "units.json"
OUT = ROOT / "data" / "advanced" / "benchmark.json"
P = "openstax-business-ethics:"


def ev(unit, kind, locator, check=None):
    page = int(re.search(r":p(\d+):", P + unit).group(1))
    return {"unit": P + unit, "page": page, "kind": kind, "locator": locator, "check": check}


DEV = [
    # ---- tables
    ("ad01", "table", "According to Table 6.2, what percentage of Germany's workforce is in unions?",
     "26%", [ev("p175:table6.2", "table", "row Germany, column 'Workforce in Unions, %' = 26", "| Germany | 26 |")]),
    ("ad02", "table", "Which country has the lowest productivity in Table 6.3 (2015), and what is its value?",
     "South Korea, 97.60 (output/hours worked)", [ev("p176:table6.3", "table", "row South Korea = 97.60 (min of column)", "| South Korea | 97.60 |")]),
    ("ad03", "table", "In the comparison of socially responsible and other index mutual funds, what was the 5-year return of the Vanguard FTSE Social Index?",
     "96.7%", [ev("p101:fig4.5", "table image", "Figure 4.5 table, row 'Vanguard FTSE Social Index', column '5-Year Return' = 96.7%")]),
    # ---- images / diagrams
    ("ad04", "image", "In the stakeholder power/interest grid, what approach is recommended for stakeholders with high power but low interest?",
     "Keep satisfied, meet needs.", [ev("p80:fig3.5", "diagram", "Figure 3.5 top-left quadrant (High power / Low interest): 'Keep satisfied, meet needs.'")]),
    ("ad05", "image", "According to the Employment Relationships figure, who pays employment taxes for non-employees such as contractors and gig workers?",
     "They must be paid entirely by the employee (the worker); for employees taxes are split between employee and employer",
     [ev("p298:fig10.8", "diagram/table image", "Figure 10.8 right column, last bullet: 'Employment taxes must be paid entirely by employee'")]),
    ("ad06", "image", "According to the map of states with right-to-work laws, which of these states has a right-to-work law: Colorado, Michigan, or New Mexico?",
     "Michigan (Colorado and New Mexico are shown without right-to-work laws)",
     [ev("p174:fig6.10", "map", "Figure 6.10: MI colored blue (with right-to-work laws); CO and NM gray")]),
    # ---- charts
    ("ad07", "chart", "What percentage of women and of men rank work-from-home options as a highly influential benefit?",
     "Women 55%, men 40%", [ev("p163:fig6.6", "bar chart", "Figure 6.6, category 'Work-from-home options': Women 55%, Men 40% (data labels)")]),
    ("ad08", "chart", "What share of the U.S. workforce is projected to be Hispanic in 2050, compared with 2010?",
     "30% projected in 2050, up from 15% in 2010", [ev("p225:fig8.2", "donut charts", "Figure 8.2 lower panel: Hispanic 15% (2010) -> 30% (projected 2050)")]),
    ("ad09", "chart", "Approximately what percentage of U.S. workers were union members in 1983?",
     "About 20%", [ev("p174:fig6.11", "line chart", "Figure 6.11: line starts at 1983 at ≈20% (y-axis gridline 20%)")]),
    ("ad10", "chart", "In the chart of employers offering telework options, which country is at 85%?",
     "China", [ev("p284:fig10.2", "donut charts", "Figure 10.2: '85% China'")]),
    # ---- multi-hop
    ("ad11", "multihop", "Which stakeholder groups shown around the company in Figure 1.2 are classified as functional stakeholders in Figure 3.4?",
     "Suppliers, employees, customers and retailers",
     [ev("p18:fig1.2", "diagram", "Figure 1.2 nodes: Clients, Customers, Suppliers, Investors, Retailers, Employees, Media, Government, Environment, Community, Competitors"),
      ev("p76:fig3.4", "diagram", "Figure 3.4 Functional Stakeholders: suppliers, employees, unions, customers, distributors, retailers")]),
    ("ad12", "multihop", "What share of millennials does the text say do not expect to stay in their current job longer than three years, and what figure does the millennials infographic give for changing jobs within the first three years?",
     "Text: 91 percent (about nine in ten); infographic: 90% change jobs within first 3 years",
     [ev("p191:t1", "text", "'About nine in ten millennials (91 percent) say they do not expect to stay...'", "(91 percent)"),
      ev("p191:fig7.2", "infographic", "Figure 7.2: '90% change jobs within first 3 years'")]),
    # ---- text
    ("ad13", "text", "When was the WARN Act enacted and how much advance notice does it require, for which events?",
     "1989; sixty days' advance notice of mass layoffs and plant closings",
     [ev("p157:t1", "text", "WARN Act of 1989 ... sixty days' advance notice of mass layoffs and plant closings", "sixty days’ advance notice of mass layoffs and plant closings")]),
    ("ad14", "text", "When were the ISO 14000 environmental standards developed and updated, and which standard covers eco-design?",
     "Developed in the 1990s, updated in 2015; ISO 14006 covers eco-design",
     [ev("p107:t1", "text", "developed in the 1990s and updated in 2015 ... eco-design (ISO 14006)", "eco-design (ISO 14006)")]),
    ("ad15", "text", "What is the CEO-to-average-worker pay multiplier in Germany, and what was the U.S. ratio in 1965?",
     "Germany: twelve; U.S. 1965: twenty to one",
     [ev("p177:t1", "text", "in Germany, it is twelve ... The 1965 U.S. ratio was only twenty to one", "in Germany, it is twelve")]),
]

DEV_UNANS = [
    ("adu1", "According to Table 6.2, what percentage of Mexico's workforce is unionized?",
     "Mexico appears only in Table 6.3 (productivity), not in Table 6.2; no Mexican union rate anywhere.",
     ["Mexico’s union", "Mexican union", "unionized in Mexico"]),
    ("adu2", "What was the 10-year return of the NASDAQ Index in the socially responsible funds comparison?",
     "Figure 4.5 only gives Year-to-Date, 2-Year and 5-Year returns.", ["10-Year Return", "10-year return", "ten-year return"]),
    ("adu3", "What percentage of employers in Japan offer telework options?",
     "Figure 10.2 covers US, China, India, UK, France, Germany only; text has no Japan telework figure.",
     ["Japan offer", "telework in Japan", "telecommuting in Japan"]),
]

TEST = [
    # ---- tables (6)
    ("at01", "table", "In Table 6.2, which country has the highest share of its workforce in unions, and what is the share?",
     "Sweden, 82%", [ev("p175:table6.2", "table", "row Sweden = 82 (max of column)", "| Sweden | 82 |")]),
    ("at02", "table", "Which country in Table 6.2 has a lower union membership rate than the United States, and what are the two rates?",
     "France, 9% (United States 12%)", [ev("p175:table6.2", "table", "rows France = 9, United States = 12", "| France | 9 |")]),
    ("at03", "table", "By how much did Germany's 2015 productivity exceed Japan's in Table 6.3?",
     "2.00 (105.90 vs 103.90 output/hours worked)", [ev("p176:table6.3", "table", "Germany 105.90 − Japan 103.90 = 2.00", "| Germany | 105.90 |")]),
    ("at04", "table", "In the comparison of socially responsible and general index funds, which index or fund had the highest year-to-date return, and how much?",
     "TIAA-CREF Social, 12.1%", [ev("p101:fig4.5", "table image", "Figure 4.5, column 'Year-to-Date Return': TIAA-CREF Social 12.1% (max)")]),
    ("at05", "table", "How many different stocks did the NASDAQ Index hold in the fund comparison table?",
     "3,176", [ev("p101:fig4.5", "table image", "Figure 4.5, row NASDAQ Index, column Holdings = 3,176")]),
    ("at06", "table", "What does the sample code of conduct require of employees regarding conflicts of interest?",
     "Employees must disclose and/or avoid any personal, financial, or other interests that might influence their ability to perform their job duties",
     [ev("p203:table7.1", "table", "row 'Conflict of interest'", "Employees must disclose and/or avoid any personal, financial, or other interests")]),
    # ---- images / diagrams (5)
    ("at07", "image", "In the classification of stakeholders around an organization, which category includes unions?",
     "Functional stakeholders", [ev("p76:fig3.4", "diagram", "Figure 3.4 box 'Functional Stakeholders' bullet 'unions'")]),
    ("at08", "image", "In the triple bottom line diagram, which three items are listed under Environmental?",
     "Land use, carbon footprint, waste", [ev("p85:fig3.6", "diagram", "Figure 3.6 right box 'Environmental': land use, carbon footprint, waste")]),
    ("at09", "image", "In the figure comparing control in Aristotle's and Confucius's virtue ethics, which behaviors are listed for Confucius?",
     "Junzi and ru (humanity)", [ev("p45:fig2.6", "diagram/table image", "Figure 2.6, row Behavior, column Confucius: junzi; ru (humanity)")]),
    ("at10", "image", "On the 2017 minimum wage by state map, which wage band is Washington state in?",
     "$10.25 – $11.50 (highest band)", [ev("p168:fig6.9", "map", "Figure 6.9: WA salmon color = legend '$10.25 – $11.50'")]),
    ("at11", "image", "According to the corporation diagram, who appoints the officers, and to what is liability limited?",
     "The directors (elected by shareholders) appoint the officers; liability is limited to corporate assets",
     [ev("p94:fig4.2", "diagram", "Figure 4.2: Shareholders → Directors → Officers arrows; 'Liability Is Limited to Corporate Assets'")]),
    # ---- charts (5)
    ("at12", "chart", "What percentage of women and of men rank better health, dental, and vision insurance as highly influential?",
     "Women 61%, men 47%", [ev("p163:fig6.6", "bar chart", "Figure 6.6 first category: Women 61%, Men 47%")]),
    ("at13", "chart", "What share of Google's technology workforce were women in the 2017 data?",
     "20% (men 80%)", [ev("p226:fig8.3", "donut chart", "Figure 8.3 'Technology' donut: 20% / 80%")]),
    ("at14", "chart", "For ethnically diverse companies, what were the likelihoods of financial performance above the industry median for the 1st and 4th diversity quartiles?",
     "1st quartile 58, 4th quartile 43 (a +35% difference)", [ev("p227:fig8.4", "bar chart", "Figure 8.4 right panel: 4th quartile 43, 1st quartile 58, '+35%'")]),
    ("at15", "chart", "In the nominal versus real minimum wage chart, in what year did the real minimum wage peak, and at roughly what value?",
     "1968, at about $11.50 (between $11 and $12)", [ev("p167:fig6.8", "line chart", "Figure 6.8 blue line max at 1968 ≈ $11.4–11.5")]),
    ("at16", "chart", "In the median net worth of families by income tier chart, in which year was upper-income median net worth highest, and roughly how high?",
     "2007, about $730K (between $700K and $750K)", [ev("p238:fig8.10", "bar chart", "Figure 8.10 Upper income bar 2007 ≈ 730K (highest)")]),
    # ---- multi-hop (4)
    ("at17", "multihop", "Is any state in the highest 2017 minimum wage band ($10.25–$11.50) also a right-to-work state?",
     "No. The highest band (Washington, California, Massachusetts and DC) are all shown without right-to-work laws",
     [ev("p168:fig6.9", "map", "Figure 6.9: WA, CA, MA (+DC) in $10.25–$11.50"),
      ev("p174:fig6.10", "map", "Figure 6.10: WA, CA, MA, DC gray = without right-to-work laws")]),
    ("at18", "multihop", "The text says the United States ranks number two on the OECD productivity list. Which country ranks above it in Table 6.3, and by how much?",
     "Canada, 109.45 vs 108.87, higher by 0.58",
     [ev("p175:t2", "text", "'number two on the list compiled by the OECD (Table 6.3)'", "number two on the list compiled by the OECD"),
      ev("p176:table6.3", "table", "Canada 109.45, United States 108.87", "| Canada | 109.45 |")]),
    ("at19", "multihop", "Among the countries that appear both in the telework chart and in Table 6.2, which has the highest share of employers offering telework, and what is its union membership rate?",
     "United States: 90% of employers offer telework; 12% of the workforce in unions",
     [ev("p284:fig10.2", "donut charts", "Figure 10.2: US 90%, UK 72%, France 71%, Germany 71% (China/India not in Table 6.2)"),
      ev("p175:table6.2", "table", "United States = 12", "| United States | 12 |")]),
    ("at20", "multihop", "Which countries appear in Table 6.3 (productivity) but not in Table 6.2 (union membership)?",
     "Mexico and South Korea",
     [ev("p176:table6.3", "table", "rows Mexico, South Korea", "| Mexico | 105.10 |"),
      ev("p175:table6.2", "table", "no Mexico / South Korea rows", "| Australia | 25 |")]),
    # ---- text (4)
    ("at21", "text", "To which companies does the Sarbanes-Oxley Act apply, who enforces it, and when was it passed?",
     "Publicly traded companies; enforced by the Securities and Exchange Commission (SEC); passed in 2002",
     [ev("p120:t3", "text", "applies to publicly traded companies and is enforced by the Securities and Exchange Commission", "applies to publicly traded companies and is enforced by the Securities and Exchange Commission"),
      ev("p120:t2", "text", "Sarbanes-Oxley, passed in 2002", "passed in 2002")]),
    ("at22", "text", "What did the court decide in Dodge v. Ford Motor Company about Ford's plan to benefit workers and customers?",
     "The court agreed with the shareholders; Ford was forced to abandon his goal of balancing profits with broader social goals",
     [ev("p96:t1", "text", "'The court agreed, and Ford was forced to abandon his managerial goal of balancing profits and realizing broader social goals.'", "The court agreed, and Ford was forced to abandon")]),
    ("at23", "text", "How much more productive were American Express employees who worked from home than office workers?",
     "43 percent more productive", [ev("p285:t3", "text", "'American Express employees who worked from home were 43 percent more productive'", "43 percent more productive")]),
    ("at24", "text", "According to the text, what results has codetermination led to in Germany?",
     "Higher productivity, fewer strikes, better pay, and safer working conditions (workers more engaged, with a real voice)",
     [ev("p175:t1", "text", "'higher productivity, fewer strikes, better pay, and safer working conditions'", "higher productivity, fewer strikes, better pay, and safer working conditions")]),
]

TEST_UNANS = [
    ("atu1", "According to Table 6.3, what was France's productivity in 2015?",
     "France is in Table 6.2 (union %) but not in Table 6.3; no French productivity value in the book.",
     ["France | ", "French productivity", "productivity in France"], ["p175:table6.2"]),
    ("atu2", "What percentage of Google's leadership roles are held by Black employees?",
     "Figure 8.3 gives leadership by gender only; text p226 gives Black share of all employees (2.4%) and White share of leadership (68%), not Black leadership.",
     ["Black leaders at Google", "Black employees hold", "leadership roles are held by Black"], []),
    ("atu3", "What was the real minimum wage in 2020 according to the nominal versus real minimum wage chart?",
     "Figure 6.8 x-axis ends around 2017; no 2020 value.", ["minimum wage in 2020", "2020 minimum wage"], []),
    ("atu4", "What percentage of women rank free lunches as a highly influential benefit?",
     "Figure 6.6 lists free coffee and free snacks, not free lunch.", ["free lunch", "Free lunch"], []),
    ("atu5", "What was Puerto Rico's minimum wage in 2017?",
     "Figure 6.9 covers 50 states + DC; Puerto Rico only mentioned for Hurricane Maria (p109).", ["Puerto Rico’s minimum", "minimum wage in Puerto Rico"], []),
    ("atu6", "What percentage of U.S. workers were union members in 1970?",
     "Figure 6.11 starts in 1983 (≈20%); no 1970 union membership rate in text.", ["union members in 1970", "1970, union", "in 1970, about"], []),
]


def main() -> None:
    units = {u["id"]: u for u in json.loads(UNITS.read_text(encoding="utf-8"))}
    items = []
    for split, rows in (("dev", DEV), ("test", TEST)):
        for qid, group, q, ans, evs in rows:
            for e in evs:
                if e["unit"] not in units:
                    raise ValueError(f"{qid}: unknown unit {e['unit']}")
                if e["check"] and e["check"] not in units[e["unit"]]["text"]:
                    raise ValueError(f"{qid}: check string not in {e['unit']}: {e['check']!r}")
                e["crop"] = units[e["unit"]].get("crop")
            items.append({"id": qid, "split": split, "group": group, "answerable": True, "question": q,
                          "expected_answer": ans, "evidence": evs, "source": "Business Ethics (OpenStax, 2026 PDF)",
                          "scoring": "pass only if answer correct+complete (units/conditions/calculation) AND citations include every required evidence unit and support the answer"})
    for split, rows in (("dev", [r + ([],) for r in DEV_UNANS]), ("test", TEST_UNANS)):
        for qid, q, reason, keywords, allowed in rows:
            hits = sorted({uid for uid, u in units.items() for k in keywords if k in u["text"]} - {P + a for a in allowed})
            if hits:
                raise ValueError(f"{qid}: absence keywords hit {hits[:3]}")
            items.append({"id": qid, "split": split, "group": "unanswerable", "answerable": False, "question": q,
                          "expected_behavior": "search_and_abstain", "absence_reason": reason,
                          "absence_check": {"keywords": keywords, "hits_outside_allowed": 0, "allowed": allowed}})
    OUT.write_text(json.dumps(items, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for split in ("dev", "test"):
        s = [i for i in items if i["split"] == split]
        print(split, len(s), {g: sum(i["group"] == g for i in s) for g in ("table", "image", "chart", "multihop", "text", "unanswerable")})


if __name__ == "__main__":
    main()
