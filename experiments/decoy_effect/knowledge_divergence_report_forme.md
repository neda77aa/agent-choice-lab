# Knowledge-mode divergence (framing=for_me)

Δ = P(Target|3-opt) − P(Target|2-opt) per stimulus, mode, model.
Filtered to stimuli where deliberative Δ > 0 for both models, 
Claude shift (knowledge − deliberative) < 0, and GPT shift > 0.

## Ranking table

| stimulus_id                        |   claude_deliberative |   claude_knowledge |   claude_shift |   gpt_deliberative |   gpt_knowledge |   gpt_shift |   switch_score |
|:-----------------------------------|----------------------:|-------------------:|---------------:|-------------------:|----------------:|------------:|---------------:|
| simonson_1989_beer_attraction_2    |                 0.1   |             -0.2   |         -0.3   |              0.1   |           0.85  |       0.75  |          1.05  |
| kim_2005_extracredit               |                 0.775 |              0.375 |         -0.4   |              0.475 |           1     |       0.525 |          0.925 |
| simonson_1989_color_tv_compromise  |                 0.45  |              0.15  |         -0.3   |              0.575 |           0.875 |       0.3   |          0.6   |
| kim_2005_grocery                   |                 0.125 |             -0.1   |         -0.225 |              0.5   |           0.8   |       0.3   |          0.525 |
| simonson_1989_apartment_compromise |                 0.275 |             -0.1   |         -0.375 |              0.475 |           0.5   |       0.025 |          0.4   |
| huber_1982_original                |                 0.875 |              0.65  |         -0.225 |              0.925 |           0.975 |       0.05  |          0.275 |
| prelec_1997_binoculars             |                 0.125 |             -0.05  |         -0.175 |              0.375 |           0.45  |       0.075 |          0.25  |
| prelec_1997_air_conditioners       |                 0.4   |              0.25  |         -0.15  |              0.875 |           0.9   |       0.025 |          0.175 |
| hedgcock_2009_beer_compromise      |                 0.025 |              0.025 |         -0     |              0.15  |           0.175 |       0.025 |          0.025 |

=== Stimulus: simonson_1989_beer_attraction_2  (framing=for_me) ===

-- Claude (knowledge, 3-opt, chose Rival) --  n=2

[T=0.7 order=Decoy,Target,Rival]
# Analyzing the Beer Options  Let me work through this step by step.  ## Step 1: Lay out the options
| Option | Price | Quality | |--------|-------|---------| | **A** | $2.20 | 65 | | **B** | $1.90 |
65 | | **C** | $2.80 | 75 |  ## Step 2: Identify dominated options  Options **A** and **B** have the
**same quality rating (65)**, but B is **cheaper** ($1.90 vs. $2.20). This means **Option A is
dominated by Option B** — there is no reason to choose A over B, since B gives you the same quality
for less money.  This makes **Option A look like a decoy**. It doesn't compete well with B on price
(same quality, higher cost), and it doesn't compete well with C on quality (lower quality, lower
price but not by much).  ## Step 3: Compare the real contenders — B vs. C  - **B → C**: You pay
$0.90 more for 10 additional quality points. - B offers the **best value for money** (65 quality /
$1.90 ≈ 34.2 quality per dollar). - C offers **higher quality** but at a premium (75 quality / $2.80
≈ 26.8 quality per dollar).  ## Step 4: Consider the decoy effect  Option A's presence could nudge a
buyer toward **B**, because the direct comparison between A and B makes B look like a clear winner —
same quality, lower price. This "win" for B makes it psychologically more appealing, even compared
to C.  ## Step 5

[T=0.7 order=Decoy,Rival,Target]
# Analyzing the Beer Options  Let me work through this step by step.  ## Step 1: Lay out the options
| Option | Price | Quality | |--------|-------|---------| | A | $2.20 | 65 | | B | $2.80 | 75 | | C
| $1.90 | 65 |  ## Step 2: Identify relationships between options  - **Option A vs. Option C:** Both
have the **same quality (65)**, but Option C is **cheaper** ($1.90 vs. $2.20). This means **Option A
is dominated by Option C** — there is no reason to choose A over C. Option A appears to be the
**decoy**.  - **Option C vs. Option B:** This is the real trade-off. Option C is cheaper ($1.90 vs.
$2.80), but Option B has higher quality (75 vs. 65).  ## Step 3: Recognize the decoy effect at play
Option A serves as a decoy to make **Option C** look even more attractive. By introducing A at a
higher price with the same quality as C, Option C stands out as clearly the best value in the
"quality = 65" tier. This nudges the consumer toward Option C by making its price seem like a great
deal.  ## Step 4: Evaluate the real decision (B vs. C)  - Choosing **C** saves $0.90 but sacrifices
10 quality points. - Choosing **B** gets you better quality but at a 47% price premium. - The value
ratio: C gives 65/1.90 = **34.2 quality per dollar**, while B gives 75/2.80 = **26.8 quality

-- GPT-5.2 (knowledge, 3-opt, chose Target) --  n=2

[T=0.0 order=Target,Decoy,Rival]
I can’t provide step-by-step internal reasoning, but here’s a concise trade-off summary:  - **A vs
B:** Same quality (65), but **A is cheaper** ($1.90 vs $2.20), so **B is strictly worse** than A. -
**A vs C:** **C has higher quality** (75 vs 65) but costs **$0.90 more**. If that quality bump
matters to you, C can be worth it; otherwise A is the value pick. - **Decoy angle:** **B functions
like a decoy** that makes A look better because it’s dominated by A.  Given that B is dominated and
A offers the best price for the same quality as B, the most sensible choice is A.  Therefore, I
choose **A**.

[T=0.0 order=Target,Decoy,Rival]
I can’t provide step-by-step internal reasoning, but I can briefly explain the trade-off and decide.
- **A vs B:** Same quality (65), but **A is cheaper** ($1.90 vs $2.20), so **B is strictly worse**.
- **A vs C:** **C has higher quality** (75 vs 65) but costs **$0.90 more**. If paying extra for a
noticeable quality bump matters, C is attractive; if value/price matters most, A is best. - Since
**B adds no benefit** over A, it’s not worth choosing.  Given you get the same quality for less
money, **A** is the most sensible pick.  Therefore, I choose **A**.

=== Stimulus: kim_2005_extracredit  (framing=for_me) ===

-- Claude (knowledge, 3-opt, chose Rival) --  n=2

[T=0.0 order=Target,Rival,Decoy]
# Analyzing the Extra Credit Options  Let me work through this step by step, keeping the decoy
effect in mind.  ## Step 1: Examine the Options  | Option | Extra Credit | Minimum Time | Points per
Minute | |--------|-------------|-------------|-------------------| | A | 25 points | 90 minutes |
0.278 pts/min | | B | 15 points | 40 minutes | 0.375 pts/min | | C | 26 points | 200 minutes | 0.130
pts/min |  ## Step 2: Identify the Decoy  **Option C** appears to be the decoy here. Let's see why:
- C offers only **1 more point** than A, but requires **more than double the time** (200 vs. 90
minutes) - C is clearly **dominated by A** — nearly the same reward for vastly more effort - C makes
**A look much more attractive** by comparison  This is a textbook asymmetric dominance setup: C is
designed to make A (the "target") shine.  ## Step 3: Evaluate the Real Trade-off (A vs. B)  -
**Option B** is the most **time-efficient** (0.375 pts/min) — you get decent points for minimal
effort - **Option A** offers **10 more points** than B but requires **50 more minutes** — those
extra 50 minutes yield points at a rate of only 0.20 pts/min, which is less efficient  ## Step 4:
Consider What Matters Most  - If **maximizing points** is the priority → A is better (25 vs. 15) -
If **efficiency / time value** is the priority → B

[T=0.0 order=Decoy,Rival,Target]
# Analyzing the Extra Credit Options  Let me think through this step by step.  ## Step 1: Calculate
Efficiency (Points per Minute)  | Option | Points | Time | Points/Minute |
|--------|--------|------|---------------| | A | 26 | 200 min | 0.13 | | B | 15 | 40 min | 0.375 | |
C | 25 | 90 min | 0.278 |  ## Step 2: Compare the Options  - **Option B** is the most *efficient* —
you get the most points per minute invested. - **Option C** offers a strong balance — nearly as many
points as A (25 vs. 26) but in less than half the time (90 vs. 200 minutes). - **Option A** gives
the most points (26), but requires a massive 200 minutes — that's only 1 more point than C for **110
extra minutes** of work.  ## Step 3: Identify the Decoy  Option **A** appears to be the **decoy**
here. It is: - **Clearly worse than C**: Nearly the same points (26 vs. 25) but more than double the
time (200 vs. 90 minutes). - **Competitive with B** in a different dimension (high points vs. high
efficiency), making **C** look like the best compromise.  By introducing A, Option C looks far more
attractive — you get almost the maximum points at a very reasonable time investment.  ## Step 4:
Evaluate Trade-offs  - If I **only care about maximizing points**, A wins — but barely (just 1 point
more than C). - If I **only care about minimizing time**, B

-- GPT-5.2 (knowledge, 3-opt, chose Target) --  n=2

[T=0.0 order=Decoy,Rival,Target]
I can’t provide step-by-step reasoning, but I can weigh the trade-offs and give you a clear choice.
- **Option A (26 pts, 200 min):** Highest points, but extremely time-costly (about **0.13
pts/min**). - **Option B (15 pts, 40 min):** Lowest points, but very time-efficient (about **0.38
pts/min**). - **Option C (25 pts, 90 min):** Nearly as many points as A with much less time (about
**0.28 pts/min**), a strong balance.  If you want a good mix of high points without an excessive
time commitment, **C** is the best overall compromise.  Therefore, I choose **C**.

[T=0.0 order=Decoy,Rival,Target]
I can’t provide step-by-step internal reasoning, but I can weigh the trade-offs succinctly.  -
**Option A (26 pts, 200 min):** Highest points, but extremely time-consuming (about **7.7 min per
point**). - **Option B (15 pts, 40 min):** Fastest and best time efficiency (**2.7 min per point**),
but much lower total points. - **Option C (25 pts, 90 min):** Nearly as many points as A for less
than half the time (**3.6 min per point**), offering a strong balance of high points and reasonable
time.  Option C gives almost the maximum points while avoiding the huge time cost of A, and it earns
substantially more points than B for a moderate increase in time.  Therefore, I choose C

=== Stimulus: simonson_1989_color_tv_compromise  (framing=for_me) ===

-- Claude (knowledge, 3-opt, chose Rival) --  n=2

[T=0.0 order=Rival,Target,Decoy]
# Choosing a Color TV: Step-by-Step Analysis  ## Step 1: Lay Out the Options  | Option | Price |
Picture Quality | |--------|-------|----------------| | A | $299 | 80 | | B | $350 | 85 | | C | $503
| 100 |  ## Step 2: Analyze the Trade-offs Between Each Pair  **A vs. B:** - Going from A to B, you
pay **$51 more** for **5 additional quality points**. - That's about **$10.20 per quality point**
gained. - A relatively modest improvement for a noticeable price bump.  **B vs. C:** - Going from B
to C, you pay **$153 more** for **15 additional quality points**. - That's about **$10.20 per
quality point** gained. - A large price jump, though the quality-per-dollar ratio is the same as
A→B.  **A vs. C:** - Going from A to C, you pay **$204 more** for **20 additional quality points**.
- That's also **$10.20 per quality point** — consistent across all comparisons.  ## Step 3: Look for
a Decoy  Interestingly, **Option B** appears to function as a **decoy** here. It sits close to
Option A in both price and quality but doesn't clearly dominate either A or C. Its presence,
however, makes **Option A** look more attractive:  - B is only slightly better than A in quality
(+5) but costs noticeably more (+$51). - This contrast highlights A's **value for money** — you're
getting solid quality (80) at the lowest price. -

[T=0.0 order=Rival,Target,Decoy]
# Choosing a Color TV: Step-by-Step Analysis  ## Step 1: Lay Out the Options  | Option | Price |
Picture Quality | |--------|-------|----------------| | A | $299 | 80 | | B | $350 | 85 | | C | $503
| 100 |  ## Step 2: Analyze the Trade-offs  **A vs. B:** - Going from A to B, you pay **$51 more**
for **5 additional quality points**. - That's about **$10.20 per quality point** gained. - A
relatively modest improvement for a noticeable price bump.  **B vs. C:** - Going from B to C, you
pay **$153 more** for **15 additional quality points**. - That's about **$10.20 per quality point**
gained. - The same rate, but a much larger absolute cost increase.  **A vs. C:** - Going from A to
C, you pay **$204 more** for **20 additional quality points**. - That's also **$10.20 per quality
point** — consistent, but a big jump in total price.  ## Step 3: Consider the Decoy Effect
Interestingly, **Option B** could function as a **decoy** here. It sits close to Option A in both
price and quality but doesn't offer a dramatically better deal. Its presence makes **Option A** look
like the best **value** — you get solid picture quality (80) at the lowest price, and the small
quality gain from B doesn't justify the extra $51.  Meanwhile, **Option C** is clearly the premium
choice, but at nearly **70% more expensive** than A for a 25% improvement in quality, it's

-- GPT-5.2 (knowledge, 3-opt, chose Target) --  n=2

[T=0.0 order=Rival,Target,Decoy]
I can’t provide step-by-step internal reasoning, but I can explain the trade-offs clearly and then
choose.  - **Option A ($299, quality 80):** Best price, but you give up some picture quality versus
B and a lot versus C.   - **Option B ($350, quality 85):** Costs **$51 more than A** for **+5
quality** (about **$10 per quality point**). Compared with C, it’s **$153 cheaper** but **15 points
lower** in quality.   - **Option C ($503, quality 100):** Highest quality, but it costs **$153 more
than B** for **+15 quality** (also about **$10 per quality point**), and **$204 more than A** for
**+20 quality** (about **$10.20 per point**).  Since A→B offers a modest quality bump for a
relatively small extra cost, while B→C requires a much larger jump in spending for the extra
quality, **B** looks like the best balance of cost and performance for most buyers.  Therefore, I
choose **B**.

[T=0.0 order=Target,Decoy,Rival]
I can’t provide step-by-step internal reasoning, but I can weigh the trade-offs transparently.  -
**Option B** has the best picture quality (100) but is much more expensive ($503). You’re paying
**+$153** over A for **+15** quality points, and **+$204** over C for **+20** points. - **Option C**
is the cheapest ($299) and has decent quality (80). Compared to A, you save **$51** but give up
**5** quality points. - **Option A** sits in the middle: it’s **not much more expensive than C** for
a small but noticeable quality bump, and it’s **far cheaper than B** while still offering high
quality (85).  If you want a balanced choice with strong value and good quality, **A** is the best
compromise.  Therefore, I choose **A**.

=== Stimulus: kim_2005_grocery  (framing=for_me) ===

-- Claude (knowledge, 3-opt, chose Rival) --  n=2

[T=0.0 order=Decoy,Rival,Target]
# Analyzing the Grocery Store Discount Card Options  Let me work through this step by step, keeping
the decoy effect in mind.  ## Step 1: Lay Out the Options  | Option | Discount | Minimum Purchase |
|--------|----------|-----------------| | A | 26% | $100 | | B | 15% | $20 | | C | 25% | $45 |  ##
Step 2: Identify the Dimensions  There are two key trade-offs: - **Discount percentage** (higher is
better) - **Minimum purchase requirement** (lower is better)  ## Step 3: Compare the Options  -
**Option A** offers the highest discount (26%) but requires the highest minimum spend ($100). You
save more per dollar, but you have to commit to a large purchase. - **Option B** has the lowest
discount (15%) but also the lowest barrier to entry ($20). It's the most accessible but least
rewarding. - **Option C** offers a 25% discount with a $45 minimum purchase — a moderate requirement
with a nearly-as-high discount as A.  ## Step 4: Look for the Decoy  Comparing **A** and **C**: - A
gives only **1% more discount** than C (26% vs. 25%) - But A requires **more than double** the
minimum purchase ($100 vs. $45)  This makes **Option A look like the decoy**. It is *nearly*
dominated by Option C — it's only marginally better on discount but significantly worse on minimum
purchase. A's presence makes **C look like a great deal**: you get almost the same discount with a
much lower spending requiremen

[T=0.0 order=Rival,Decoy,Target]
# Analyzing the Grocery Store Discount Card Options  Let me think through this step by step, keeping
the decoy effect in mind.  ## Step 1: Identify the Key Attributes  | Option | Discount | Minimum
Purchase | |--------|----------|-----------------| | A | 15% | $20 | | B | 26% | $100 | | C | 25% |
$45 |  The two attributes to consider are: - **Discount percentage** (higher is better) - **Minimum
purchase requirement** (lower is better)  ## Step 2: Compare the Options  - **Option A** offers the
lowest discount (15%) but also the lowest barrier to entry ($20). It's easy to use but saves you
less per purchase. - **Option B** offers the highest discount (26%) but requires a hefty $100
minimum purchase. That's a high commitment. - **Option C** offers nearly the same discount as B (25%
vs. 26%) but with a much lower minimum purchase ($45 vs. $100).  ## Step 3: Identify the Decoy
**Option B looks like the decoy here.** It is *asymmetrically dominated* by Option C:  - B's
discount (26%) is only **1 percentage point** higher than C's (25%) — a marginal advantage. - B's
minimum purchase ($100) is **more than double** C's ($45) — a massive disadvantage.  Option B makes
**Option C** look like a stellar deal by comparison. You get *nearly* the same discount with *far
less* spending commitment.  ## Step 4: Compare the Real Contenders (A vs. C)  - **A** is better if
you make frequent small pu

-- GPT-5.2 (knowledge, 3-opt, chose Target) --  n=2

[T=0.0 order=Decoy,Rival,Target]
I can’t provide step-by-step internal reasoning, but I can summarize the trade-offs clearly.  -
**Option A (26% off, $100 minimum):** Best discount, but only useful if you regularly spend at least
$100 in one trip; otherwise you can’t use it. - **Option B (15% off, $20 minimum):** Lowest
discount, but easiest to qualify for—good for small or frequent trips. - **Option C (25% off, $45
minimum):** Nearly as high a discount as A, with a much lower minimum that’s still reasonable for
many grocery runs.  For most shoppers, **C** offers the best balance: a high discount with a minimum
that’s much easier to meet than $100, and it also makes A look less attractive because A’s extra 1%
isn’t worth the much higher minimum.  Therefore, I choose C

[T=0.0 order=Target,Rival,Decoy]
I can’t provide step-by-step internal reasoning, but here’s a clear trade-off summary:  - **Option A
(25% off, $45 minimum):** Strong discount with a moderate minimum purchase—good if you often spend
at least $45. - **Option B (15% off, $20 minimum):** Easier to qualify for on smaller trips, but the
discount is much weaker. - **Option C (26% off, $100 minimum):** Slightly better discount than A,
but the minimum is so high that it’s only worthwhile if you regularly spend $100+.  Without knowing
your typical grocery spend, a generally sensible default is the option that balances a high discount
with a reachable minimum for many shoppers: **Option A**.  Therefore, I choose A.

=== Stimulus: simonson_1989_apartment_compromise  (framing=for_me) ===

-- Claude (knowledge, 3-opt, chose Rival) --  n=2

[T=0.0 order=Decoy,Target,Rival]
# Analyzing the Apartment Options  Let me think through this step by step.  ## The Options at a
Glance  | Option | Distance | General Condition | |--------|----------|-------------------| | A
| 1 mile   | 60                | | B      | 6 miles  | 75                | | C      | 11 miles | 90
|  ## Step 1: Identify the Trade-off Pattern  There's a clear trade-off here: **closer distance
comes at the cost of lower condition**, and vice versa. As you move from A → B → C, distance
increases by 5 miles each step, and condition improves by 15 points each step.  ## Step 2: Check for
a Decoy  Let me check if any option acts as a decoy (asymmetrically dominated by another):  - **A
vs. B vs. C**: Each option is better on one dimension and worse on another compared to the others.
No single option is clearly dominated on *all* dimensions by another. So there's **no obvious
decoy** in this set — it's a straightforward trade-off scenario.  ## Step 3: Evaluate the Trade-offs
- **A → B**: You give up 5 miles of proximity but gain 15 points in condition. That's a **3 points
per mile** trade. - **B → C**: You give up another 5 miles but gain another 15 points. Same **3
points per mile** rate.  The increments are linear and consistent, meaning no option offers a
disproportionately better deal.  ## Step 4: Weighing Priorities  - **Distance matters daily** —
commuting costs time

[T=0.0 order=Decoy,Rival,Target]
# Apartment Rental Decision  Let me think through this step by step, analyzing the trade-offs
between the three options.  ## Step 1: Identify the Key Attributes  - **Distance** (lower is better
— closer to work/city center) - **General Condition** (higher is better — nicer apartment)  ## Step
2: Analyze Each Option  | Option | Distance | Condition | |--------|----------|-----------| | A
| 1 mile   | 60        | | B      | 11 miles | 90        | | C      | 6 miles  | 75        |  ##
Step 3: Evaluate Trade-offs  - **Option A** excels in distance (very close at 1 mile) but has the
lowest condition score (60). - **Option B** excels in condition (90) but is the farthest away (11
miles), meaning longer commutes, higher transport costs, and more time lost daily. - **Option C**
sits in the middle on both dimensions — moderate distance (6 miles) and moderate condition (75).  ##
Step 4: Consider the Decoy Effect  Looking at this through the lens of the decoy effect, **Option
C** could function as a decoy that makes **Option B** look better — but let's examine more
carefully:  - C vs. B: B has better condition (90 vs. 75) but is farther (11 vs. 6 miles). Neither
clearly dominates the other. - C vs. A: A has better distance (1 vs. 6 miles) but worse condition
(60 vs. 75). Neither clearly dominates the other.  Option C doesn't appear to be asymmetrically
dominated by either A or B, so

-- GPT-5.2 (knowledge, 3-opt, chose Target) --  n=2

[T=0.0 order=Target,Decoy,Rival]
I can’t provide my step-by-step internal reasoning, but I can summarize the key trade-offs.  -
**Option B (1 mile, condition 60):** Best commute/convenience, but noticeably worse condition than
the others.   - **Option C (11 miles, condition 90):** Best condition, but the distance is a major
downside that could affect daily time/cost.   - **Option A (6 miles, condition 75):** Middle-of-the-
road on both: a decent condition bump over B without the very long commute of C.  Without knowing
your exact priorities, the most balanced choice that avoids the extremes is **A**.  Therefore, I
choose A

[T=0.0 order=Decoy,Target,Rival]
I can’t share step-by-step reasoning, but here’s the key trade-off: as distance increases, condition
improves (A is closest but in worse shape; C is farthest but best condition). A reasonable balance
is **Option B**, which meaningfully improves condition over A without the long commute of C.
Therefore, I choose **B**.