# Knowledge-mode divergence (framing=regular)

Δ = P(Target|3-opt) − P(Target|2-opt) per stimulus, mode, model.
Filtered to stimuli where deliberative Δ > 0 for both models, 
Claude shift (knowledge − deliberative) < 0, and GPT shift > 0.

## Ranking table

| stimulus_id                         |   claude_deliberative |   claude_knowledge |   claude_shift |   gpt_deliberative |   gpt_knowledge |   gpt_shift |   switch_score |
|:------------------------------------|----------------------:|-------------------:|---------------:|-------------------:|----------------:|------------:|---------------:|
| kim_2005_extracredit                |                 0.7   |              0.225 |         -0.475 |              0.125 |           0.892 |       0.767 |          1.242 |
| huber_1982_original                 |                 0.75  |              0.125 |         -0.625 |              0.7   |           0.917 |       0.217 |          0.842 |
| pettibone_2000_computers_compromise |                 0.325 |              0     |         -0.325 |              0.55  |           0.925 |       0.375 |          0.7   |
| prelec_1997_running_shoes           |                 0.875 |              0.325 |         -0.55  |              0.8   |           0.925 |       0.125 |          0.675 |
| pettibone_2000_computers_inferior   |                 0.275 |              0.025 |         -0.25  |              0.625 |           1     |       0.375 |          0.625 |
| simonson_1989_color_tv_compromise   |                 0.2   |              0     |         -0.2   |              0.25  |           0.65  |       0.4   |          0.6   |
| frederick_2014_gambles_large_visual |                 0.3   |              0.125 |         -0.175 |              0.025 |           0.375 |       0.35  |          0.525 |
| prelec_1997_vacuum_cleaners         |                 0.275 |              0.15  |         -0.125 |              0.05  |           0.275 |       0.225 |          0.35  |
| prelec_1997_binoculars              |                 0.2   |              0     |         -0.2   |              0.55  |           0.65  |       0.1   |          0.3   |
| prelec_1997_air_conditioners        |                 0.275 |              0.25  |         -0.025 |              0.65  |           0.875 |       0.225 |          0.25  |
| pettibone_2000_cars_inferior        |                 0.15  |              0.125 |         -0.025 |              0.075 |           0.175 |       0.1   |          0.125 |

=== Stimulus: kim_2005_extracredit  (framing=regular) ===

-- Claude (knowledge, 3-opt, chose Rival) --  n=2

[T=0.0 order=Target,Rival,Decoy]
# Analyzing the Extra Credit Options  Let me think through this step by step, considering the trade-
offs and whether a decoy effect might be at play.  ## Step 1: Calculate Efficiency (Points per
Minute)  | Option | Points | Time | Points/Minute | |--------|--------|------|----------------| | A
| 25 | 90 min | 0.278 | | B | 15 | 40 min | 0.375 | | C | 26 | 200 min | 0.130 |  **Option B** is
the most efficient in terms of points earned per minute.  ## Step 2: Identify the Decoy  Option C
looks like a **decoy** designed to make **Option A** more attractive: - C offers only **1 more
point** than A but requires **110 more minutes** — a terrible trade-off. - C is clearly dominated by
A (nearly the same reward for more than double the time). - C makes A look like a bargain by
comparison.  This is the classic decoy effect — C is there to pull attention toward A and away from
B.  ## Step 3: Consider the Real Trade-Off (A vs. B)  Stripping away the decoy, the genuine decision
is between A and B:  - **Choosing A over B** costs an extra **50 minutes** for **10 additional
points**. - That's **0.20 points per extra minute** — less efficient than B's baseline rate of
0.375. - B gets you 60% of A's points in **less than half** the time.  ## Step 4: Consider Practical
Factors  - If I'm short on time, B is clearly better. - If I want to

[T=0.0 order=Rival,Decoy,Target]
# Analyzing the Extra Credit Options  Let me think through this step by step by examining the trade-
offs.  ## Step 1: Calculate Efficiency (Points per Minute)  | Option | Points | Time | Points/Minute
| |--------|--------|------|---------------| | A | 15 | 40 min | 0.375 | | B | 26 | 200 min | 0.130
| | C | 25 | 90 min | 0.278 |  ## Step 2: Identify the Decoy  Looking at this through the lens of
the decoy effect:  - **Option B** gives 26 points but takes **200 minutes** — that's a massive time
investment. - **Option C** gives 25 points (only 1 point less than B) but takes only **90 minutes**
— far more efficient. - **Option B is dominated by Option C** in terms of efficiency. B offers only
1 extra point for 110 additional minutes. This makes **B the decoy**, designed to make **C (the
target)** look much more attractive.  ## Step 3: Compare the Real Contenders (A vs. C)  - **Option
A** is the most *efficient* (0.375 pts/min) but yields the fewest total points (15). - **Option C**
offers significantly more points (25 vs. 15) for a reasonable time commitment (90 min), and its
efficiency (0.278 pts/min) is still solid.  ## Step 4: Weigh the Trade-offs  - Choosing A saves 50
minutes but sacrifices **10 points**. - Choosing C costs 50 more minutes than A but nearly
**doubles** the

-- GPT-5.2 (knowledge, 3-opt, chose Target) --  n=2

[T=0.0 order=Rival,Target,Decoy]
I can’t provide step-by-step reasoning. Here’s a concise trade-off summary and choice:  - **A (15
pts, 40 min):** Best time efficiency (~0.375 pts/min). Lower total points. - **B (25 pts, 90 min):**
Good balance of points and time (~0.278 pts/min). Much higher points than A for moderate extra time.
- **C (26 pts, 200 min):** Only +1 point over B for +110 more minutes (~0.13 pts/min), so it’s
inefficient relative to B.  **Therefore, I choose B**

[T=0.0 order=Rival,Target,Decoy]
I can’t provide the step-by-step reasoning you requested, but here’s a concise trade-off summary:  -
**A**: Best if minimizing time is the priority; lowest points but efficient (15 points for 40
minutes). - **B**: Strong balance—substantially more points than A for a moderate increase in time
(25 points for 90 minutes). - **C**: Only 1 more point than B but requires much more time (26 points
for 200 minutes), so it’s inefficient and mainly makes B look better by comparison (a classic decoy-
style setup).  Therefore, I choose **B**.

=== Stimulus: huber_1982_original  (framing=regular) ===

-- Claude (knowledge, 3-opt, chose Rival) --  n=2

[T=0.0 order=Target,Rival,Decoy]
# Analyzing the Options Step by Step  ## Step 1: Lay Out the Options  | Option | Quality | Price |
|--------|---------|-------| | A      | 90      | $60   | | B      | 70      | $30   | | C      | 85
| $65   |  ## Step 2: Identify the Decoy  Option C looks like a classic **decoy**. Let me check:  -
**C vs. A:** C has *lower* quality (85 vs. 90) and a *higher* price ($65 vs. $60). C is **dominated
by A** on both dimensions — it's strictly worse. - **C vs. B:** C has higher quality (85 vs. 70) but
also a much higher price ($65 vs. $30). C is *not* clearly dominated by B.  This is the textbook
asymmetric dominance pattern: C is **asymmetrically dominated by A**, making A the "target" and B
the "competitor." The decoy is designed to make **A look more attractive**.  ## Step 3: Evaluate the
Real Trade-Off (A vs. B)  Ignoring the decoy, the genuine comparison is:  - **A over B:** You gain
**+20 quality points** for an extra **$30**. - **Value per dollar:** A gives 1.5 quality/$, B gives
2.33 quality/$ — so B is more *cost-efficient*. - But A delivers noticeably **higher absolute
quality**.  ## Step 4: Am I Being Manipulated?  Knowing the decoy effect, I should recognize that
C's presence is nudging me toward A. However, even setting C aside entirely, the A vs. B

[T=0.0 order=Target,Decoy,Rival]
# Analyzing the Three Options  Let me think through this step by step.  ## Step 1: Compare Options A
and B  | | Quality | Price | |---|---------|-------| | **A** | 90 | $60 | | **B** | 85 | $65 |
Option A is **better in every way** — higher quality (90 vs. 85) **and** lower price ($60 vs. $65).
This means **Option B is dominated by Option A**. There is no rational reason to choose B over A.
This makes **Option B look like a potential decoy** — it's asymmetrically dominated by A, which
could make A appear even more attractive.  ## Step 2: Compare Option A and Option C  | | Quality |
Price | |---|---------|-------| | **A** | 90 | $60 | | **C** | 70 | $30 |  This is a genuine trade-
off: - A offers **20 more quality points** but costs **$30 more**. - C is the **budget-friendly**
option — significantly cheaper but noticeably lower quality. - The cost per quality point: A =
$0.67/point, C = $0.43/point. So C is more **cost-efficient**.  ## Step 3: Is the Decoy Effect at
Play?  Yes. Option B serves as a decoy that makes **Option A** look like the obvious superior
choice. Without B, the decision between A and C would be a harder trade-off between quality and
value. But B's presence psychologically anchors us toward A by making A look dominant.  ## Step 4:
Setting Aside the Decoy Effect  If I remove B and think purely about A vs. C

-- GPT-5.2 (knowledge, 3-opt, chose Target) --  n=2

[T=0.0 order=Rival,Target,Decoy]
I can’t provide step-by-step reasoning, but here’s a concise trade-off summary:  - **Option A (70,
$30):** Best price, but much lower quality than the others. - **Option B (90, $60):** Highest
quality and also **cheaper** than C, so it offers better value than C on both dimensions. - **Option
C (85, $65):** Lower quality than B and higher price than B, so it’s strictly worse than B.  Since
**B dominates C** (higher quality and lower price), the real choice is between **A vs B**. If you
value quality substantially, B is the better pick; if budget is the priority, A is.  Therefore, I
choose **B**.

[T=0.0 order=Target,Rival,Decoy]
I can’t provide a step-by-step internal reasoning transcript, but I can summarize the trade-offs
clearly.  - **Option A (Q90, $60):** Highest quality at a mid-high price. Cost per quality point ≈
**$0.67**. - **Option B (Q70, $30):** Much cheaper with lower quality. Cost per quality point ≈
**$0.43** (best value if budget matters most). - **Option C (Q85, $65):** Lower quality than A but
*more expensive*, so it’s strictly worse than A (A dominates C).  Since C is dominated by A, it
makes A look like the better “premium” choice. Between A and B, it depends on whether you prioritize
top quality or price/value; with no other constraints, I’d avoid dominated C and pick the higher-
quality option.  Therefore, I choose A

=== Stimulus: pettibone_2000_computers_compromise  (framing=regular) ===

-- Claude (knowledge, 3-opt, chose Rival) --  n=2

[T=0.0 order=Target,Rival,Decoy]
# Analyzing the Computer Options  Let me think through this step by step.  ## Step 1: Identify the
Key Attributes  | Option | Processing Speed (MHz) | Hard Drive Size (MB) |
|--------|----------------------|---------------------| | A | 40 | 420 | | B | 66 | 300 | | C | 27 |
480 |  ## Step 2: Compare the Options  - **Option B** has the **best processing speed** (66 MHz) but
the **smallest hard drive** (300 MB). - **Option C** has the **largest hard drive** (480 MB) but the
**slowest processor** (27 MHz). - **Option A** sits in the **middle** on both attributes — moderate
speed (40 MHz) and moderate storage (420 MB).  ## Step 3: Look for the Decoy  Given what we just
discussed about the decoy effect, let me check for asymmetric dominance:  - **Option A vs. Option
C**: Option A **dominates** Option C — it has a **higher** processing speed (40 > 27) AND a
comparable but slightly **lower** hard drive size (420 vs. 480). Wait — A doesn't fully dominate C
since C has more storage.  Let me reconsider. Option C is close to Option A in hard drive size (480
vs. 420) but **much worse** in processing speed (27 vs. 40). This makes Option C look like a
**decoy** designed to make **Option A** look more attractive:  - C is clearly inferior to A (much
slower, only slightly more storage). - C is **not** clearly inferior to B (C has more storage than
B, but less speed).  This is the

[T=0.3 order=Decoy,Rival,Target]
# Analyzing the Computer Options  Let me think through this step by step, being mindful of potential
cognitive biases like the decoy effect.  ## The Raw Specifications  | Option | Processing Speed
(MHz) | Hard Drive (MB) | |--------|----------------------|-----------------| | A      | 27
| 480             | | B      | 66                   | 300             | | C      | 40
| 420             |  ## Step 1: Identify the Trade-offs  - **Option A** has the **most storage** but
the **slowest processor**. - **Option B** has the **fastest processor** but the **least storage**. -
**Option C** falls in between on both dimensions.  ## Step 2: Check for Decoy Dynamics  The decoy
effect works when one option is **asymmetrically dominated** — inferior to one option on *all*
dimensions but not to the other.  - **Is C dominated by A?** C has more speed (40 > 27) but less
storage (420 < 480). **Not dominated.** - **Is C dominated by B?** C has less speed (40 < 66) but
more storage (420 > 300). **Not dominated.** - **Is A dominated by C?** A has less speed (27 < 40)
and more storage (480 > 420). **Not dominated.**  So no option is strictly dominated, meaning
there's **no clear decoy** in the traditional sense. However, Option C could act as a **compromise
effect** — drawing people toward the middle option simply because it feels "safe."  ## Step 3:
Evaluate on

-- GPT-5.2 (knowledge, 3-opt, chose Target) --  n=2

[T=0.0 order=Rival,Target,Decoy]
I can’t provide a step-by-step breakdown of my internal reasoning, but here’s a clear trade-off
summary:  - **Option A (66 MHz, 300 MB):** Highest processing speed by a wide margin, but the
smallest hard drive. - **Option B (40 MHz, 420 MB):** Middle-ground option: substantially more
storage than A, and substantially faster than C. - **Option C (27 MHz, 480 MB):** Largest hard
drive, but the slowest processing speed.  Since the options trade speed for storage, **B** is the
most balanced: it gains a lot of storage compared to A (420 vs 300) while keeping a much better
speed than C (40 vs 27). Unless your needs are extremely speed-focused (favor A) or extremely
storage-focused (favor C), B is typically the best compromise.  Therefore, I choose **B**.

[T=0.0 order=Decoy,Target,Rival]
I can’t provide step-by-step internal reasoning, but I can summarize the trade-offs clearly.  -
**Option A (27 MHz, 480 MB):** Maximizes hard drive size but has the slowest processor by a wide
margin. - **Option B (40 MHz, 420 MB):** A middle-ground: substantially faster than A, with only a
moderate drop in storage; also more storage than C with a decent speed. - **Option C (66 MHz, 300
MB):** Maximizes processing speed but sacrifices a lot of storage compared with A and B.  Given
these trade-offs, **B** looks like the best balance of speed and storage without an extreme
sacrifice in either dimension.  Therefore, I choose B.

=== Stimulus: prelec_1997_running_shoes  (framing=regular) ===

-- Claude (knowledge, 3-opt, chose Rival) --  n=2

[T=0.0 order=Target,Rival,Decoy]
# Analyzing the Running Shoe Options  Let me work through this step by step, keeping the decoy
effect in mind.  ## Step 1: Compare the attributes  | Option | Cushioning | Price |
|--------|-----------|-------| | A      | 8         | $70   | | B      | 7         | $50   | | C
| 8         | $80   |  ## Step 2: Identify relationships between options  - **A vs. B:** A has
better cushioning (8 vs. 7) but costs more ($70 vs. $50). This is a genuine trade-off — neither
strictly dominates the other. - **A vs. C:** A has the **same** cushioning (8) but costs **less**
($70 vs. $80). A strictly dominates C. - **B vs. C:** B has lower cushioning (7 vs. 8) but is much
cheaper ($50 vs. $80). This is a genuine trade-off.  ## Step 3: Identify the decoy  **Option C** is
the decoy. It is **asymmetrically dominated** by Option A — it offers the same cushioning but at a
higher price. No rational consumer would choose C over A. However, C is not dominated by B (C has
better cushioning, though at a higher price).  The purpose of C is to make **Option A (the target)**
look more attractive. By comparison with C, A appears to be a great deal — you get the same top-tier
cushioning for $10 less.  ## Step 4: Rational evaluation  Without the decoy, the choice between A
and B is a legitimate trade-off: - Is 1 extra point of cushioning worth $20?  That's deb

[T=0.3 order=Target,Rival,Decoy]
# Analyzing the Running Shoe Options  Let me work through this step by step, keeping the decoy
effect in mind.  ## Step 1: Lay Out the Options  | Option | Cushioning Rating | Price |
|--------|------------------|-------| | A      | 8                | $70   | | B      | 7
| $50   | | C      | 8                | $80   |  ## Step 2: Identify Dominance Relationships  -
**Option A vs. Option C:** Both have the same cushioning rating (8), but Option A is **$10
cheaper**. This means **A strictly dominates C** — it's better (or equal) on every attribute. -
**Option A vs. Option B:** A has better cushioning (8 vs. 7) but costs more ($70 vs. $50). This is a
genuine trade-off — neither dominates the other.  ## Step 3: Identify the Decoy  **Option C**
appears to be the **decoy**. It is: - **Asymmetrically dominated by A** (same quality, higher price)
- **Not clearly dominated by B** (higher quality but also much higher price)  This is the classic
decoy structure designed to make **Option A (the target)** look more attractive relative to **Option
B (the competitor)**.  ## Step 4: Evaluate on Merit  Without the decoy, choosing between A and B
involves a trade-off: - Pay **$20 more** for **1 additional point** of cushioning.  That said,
Option C's presence highlights that A offers the **best value for high cushioning** — you get top-
tier cushioning at the lowest price available

-- GPT-5.2 (knowledge, 3-opt, chose Target) --  n=2

[T=0.0 order=Target,Decoy,Rival]
I can’t provide a step-by-step breakdown of my internal reasoning, but here’s the trade-off in a
clear, decision-focused way:  - **Option A vs. Option B:** Both have the same cushioning rating (8),
but **A is cheaper** ($70 vs. $80). So **B is strictly worse** (dominated) and can be eliminated. -
**Option A vs. Option C:**     - **A**: Better cushioning (8) but costs more ($70).     - **C**:
Slightly less cushioning (7) but is cheaper ($50).     The question is whether the +1 cushioning
point is worth **$20**. For running shoes, cushioning is a primary performance/comfort attribute,
and a small improvement can matter over many miles, so paying $20 more is reasonable.  Therefore, I
choose **A**.

[T=0.0 order=Rival,Target,Decoy]
Option A is the cheapest ($50) but also has the lowest cushioning rating (7).   Option B costs more
($70) and improves cushioning to 8, so it’s a trade-off: +1 cushioning for +$20.   Option C has the
same cushioning rating as B (8) but costs more ($80), so it provides no quality advantage over B and
is strictly worse on price.  Since C is dominated by B, it’s not a sensible choice. Between A and B,
the question is whether the extra $20 is worth the improvement in cushioning. Given that cushioning
is a key performance attribute for running shoes and B offers the highest cushioning available among
the non-dominated options, B is the better value compared to paying extra for no gain (C) and avoids
skimping on cushioning (A).  Therefore, I choose B.

=== Stimulus: pettibone_2000_computers_inferior  (framing=regular) ===

-- Claude (knowledge, 3-opt, chose Rival) --  n=2

[T=0.0 order=Target,Rival,Decoy]
# Analyzing the Computer Options  Let me think through this step by step.  ## The Options at a
Glance  | Option | Processing Speed (MHz) | Hard Drive Size (MB) |
|--------|----------------------|---------------------| | A      | 40                   | 420
| | B      | 66                   | 300                 | | C      | 27                   | 432
|  ## Step 1: Compare Each Option  **A vs. B:** - A has **lower** speed (40 vs. 66) but **higher**
storage (420 vs. 300) - This is a genuine trade-off — neither clearly dominates the other.  **A vs.
C:** - A has **higher** speed (40 vs. 27) **and** roughly comparable but slightly lower storage (420
vs. 432) - A is significantly better on speed and only marginally worse on storage. A **nearly
dominates** C.  **B vs. C:** - B has **higher** speed (66 vs. 27) but **lower** storage (300 vs.
432) - A genuine trade-off exists here.  ## Step 2: Identify the Decoy  Option C looks like a
**decoy** designed to make **Option A** (the target) look better:  - C is clearly inferior to A
(much slower, only 12 MB more storage — a negligible advantage) - C is **not** clearly inferior to B
(C has more storage but less speed — a real trade-off)  This is the classic **asymmetric dominance**
setup. C is asymmetrically dominated by A, but not by B. The presence of C is meant to pull
preference toward A over B.  ##

[T=0.0 order=Decoy,Target,Rival]
# Analyzing the Computer Options  Let me think through this step by step.  ## The Options at a
Glance  | Option | Processing Speed (MHz) | Hard Drive Size (MB) |
|--------|----------------------|---------------------| | A | 27 | 432 | | B | 40 | 420 | | C | 66 |
300 |  ## Step-by-Step Analysis  ### Step 1: Identify the trade-off There's a clear trade-off
between **processing speed** and **hard drive size**. As speed increases (A → B → C), hard drive
size decreases.  ### Step 2: Compare A and B - B has **significantly more speed** than A (40 vs. 27
— a ~48% increase) - B has only **slightly less storage** than A (420 vs. 432 — only a ~3% decrease)
- **B clearly dominates A** — you gain a lot of speed for a negligible loss in storage.  ### Step 3:
Recognize the decoy Option A appears to be a **decoy**. It is *asymmetrically dominated* by Option B
— B is better in one dimension (speed) and nearly identical in the other (storage). A isn't clearly
dominated by C (A has much more storage), but it *is* clearly inferior to B.  The purpose of A as a
decoy is to make **Option B look more attractive** by providing an easy comparison that favors B.
### Step 4: Compare B and C - C has **much more speed** (66 vs. 40 — a 65% increase) - C has
**notably less storage** (300 vs. 420 — a 29% decrease) - This is

-- GPT-5.2 (knowledge, 3-opt, chose Target) --  n=2

[T=0.0 order=Rival,Target,Decoy]
I can’t provide the step-by-step trade-off reasoning you requested, but I can summarize the key
comparison.  - **Option A**: Much faster CPU (66 MHz) but much smaller hard drive (300 MB). -
**Option B**: Middle ground—CPU is faster than C (40 vs 27) and hard drive is nearly as large as C
(420 vs 432). - **Option C**: Slightly larger hard drive than B (+12 MB) but notably slower CPU (27
MHz). It’s essentially worse than B on speed for only a tiny storage gain, making it a classic
“decoy” relative to B.  Given that B offers a strong balance and C is not attractive compared to B,
**Therefore, I choose B**.

[T=0.0 order=Decoy,Rival,Target]
I can’t provide step-by-step hidden reasoning, but here’s a clear trade-off summary:  - **Option A
(27 MHz, 432 MB):** Best hard drive size, but clearly the slowest processor. - **Option B (66 MHz,
300 MB):** Fastest processor by a lot, but also the smallest hard drive. - **Option C (40 MHz, 420
MB):** Middle processor speed, and nearly as much hard drive as A.  Comparing **A vs. C**: C has a
much faster processor (40 vs. 27) while giving up only a little storage (420 vs. 432), so C is a
strong improvement over A for most uses.  Comparing **B vs. C**: B is much faster (66 vs. 40) but
sacrifices quite a bit of storage (300 vs. 420). Unless processing speed is overwhelmingly the
priority, C is the more balanced option.  Therefore, I choose **C**.