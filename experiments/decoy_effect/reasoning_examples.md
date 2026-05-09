# Decoy reasoning examples — Claude & GPT-5.2 (3-opt only)

Filter: deliberative + knowledge modes, both framings. Avoidance = chose Rival AND text matches decoy/manipulation regex. Falling = chose Target AND text doesn't mention decoy/manipulation.


## AVOIDING decoy — Claude (deliberative)

### Claude | deliberative | pettibone_2000_cdplayers_phantom | choice=Rival

```
I need to choose between the available options, keeping in mind that Option A is currently unavailable. Let me analyze the available options: **Option B:** $250, 10 disks **Option C:** $175, 4 disks Let me think about the trade-offs: - **Price difference:** Option C is $75 cheaper than Option B ($175 vs $250). - **Disk capacity difference:** Option B holds 6 more disks than Option C (10 vs 4). - **Cost per disk:** Option B = $25/disk, Option C = $43.75/disk. The key consideration is whether the additional 6 disks of capacity are worth the extra $75. That's about $12.50 per additional disk. Option B offers significantly more capacity (2.5x more disks) for a moderate price increase. For a CD player, having 10 disks versus 4 disks is a substantial convenience improvement, as you'd need to change disks much less frequently. However, I should also consider the role of Option A being mentioned but unavailable. Option A was a better deal than Option B (same price, 3 more disks). Its mention might be designed to make Option B seem less attractive by comparison. Without Option A's influence, the straightforward comparison between B and C should stand on its own merits. Option C offers solid value at a lower price point, saving $75. For someone who doesn't need the extra capacity, this is a practical choice. The 4-disk capacity is functional, and the savings are meaningful. Weighing the  …
```

### Claude | deliberative | pettibone_2000_computers_phantom | choice=Rival

```
I need to choose between the available options A and C (since Option B is unavailable). Let me compare them: **Option A:** 40 Mhz processing speed, 420 MB hard drive **Option C:** 66 Mhz processing speed, 300 MB hard drive **Trade-offs:** - **Processing Speed:** Option C has significantly faster processing speed (66 Mhz vs. 40 Mhz) — that's a 65% improvement. This is a substantial advantage for performance. - **Hard Drive Size:** Option A has a larger hard drive (420 MB vs. 300 MB) — that's 120 MB more storage, a 40% advantage. **Analysis:** This is a classic trade-off between speed and storage. Processing speed affects how fast the computer runs applications and handles tasks, while hard drive size determines how much data and software you can store. Now, the interesting aspect of this problem is the "phantom decoy" effect. Option B (40 Mhz, 480 MB) is unavailable but listed. Option B would have been clearly better than Option A on hard drive size while having the same processing speed. The presence of Option B — even though unavailable — might psychologically draw attention to the hard drive dimension and make Option A seem more relevant along that attribute. However, thinking rationally about the actual available choices: Option C offers a much faster processor (66 vs. 40 Mhz), which is generally a more critical performance factor. The hard drive trade-off of 120 MB less sto …
```


## AVOIDING decoy — Claude (knowledge)

### Claude | knowledge | pettibone_2000_planetickets_phantom | choice=Rival

```
# Analyzing the Plane Ticket Options Let me work through this step by step. ## First: Option A is unavailable Option A ($400, 15 min layover) is explicitly stated as **unavailable**, so I cannot choose it regardless of how attractive it might be. ## Remaining Options | Option | Cost | Layover | |--------|------|---------| | **B** | $330 | 150 min (2.5 hours) | | **C** | $400 | 60 min (1 hour) | ## Analyzing Trade-offs - **Option B** saves $70 but comes with a 150-minute layover. - **Option C** costs $70 more but has a significantly shorter layover of 60 minutes — that's 90 minutes less waiting. ## Recognizing the Decoy Effect This is a great opportunity to apply what we just discussed. **Option A** — even though unavailable — could serve as a psychological anchor/decoy. It was priced at $400 with only a 15-minute layover, which might make Option C ($400, 60 min) look worse by comparison (same price, longer layover). However, since A is unavailable, the real comparison is only between B and C. ## Evaluating on Merits - **$70 savings vs. 90 fewer minutes of layover**: This comes down to how much I value my time. $70 for 90 minutes of saved time works out to roughly **$0.78 per minute** or about **$47/hour**. - A 150-minute layover is quite long and can be tiring and inconvenient. - A 60-minute layover is reasonable and comfortable — enough time to make a connection without excess …
```

### Claude | knowledge | prelec_1997_vacuum_cleaners | choice=Rival

```
# Analyzing the Vacuum Cleaner Options Let me think through this step by step. ## Step 1: Understand the trade-offs | Option | Suction Power | Price | Cost per unit of suction | |--------|--------------|-------|--------------------------| | **A** | 70 | $150 | $2.14/unit | | **B** | 60 | $100 | $1.67/unit | | **C** | 80 | $200 | $2.50/unit | ## Step 2: Compare the options pairwise - **B vs. A:** Going from B to A, you pay $50 more for 10 extra suction points — that's $5 per extra unit of suction. - **A vs. C:** Going from A to C, you pay $50 more for 10 extra suction points — also $5 per extra unit of suction. - **B vs. C:** Going from B to C, you pay $100 more for 20 extra suction points — $5 per extra unit. ## Step 3: Check for a decoy The incremental cost per unit of suction is the same across all jumps ($5/unit), so none of these options is clearly *dominated* by another. There's no obvious decoy here — each option represents a consistent trade-off between price and performance. ## Step 4: Evaluate value - **Option B** has the best *value ratio* (lowest cost per unit of suction at $1.67/unit). - **Option A** is a middle-ground option — moderate power at a moderate price. - **Option C** offers the highest performance but at the highest price and worst value ratio. ## Step 5: Make a decision Since none of the options is dominated, this comes down to how much I value additiona …
```


## AVOIDING decoy — GPT-5.2 (deliberative)

### GPT-5.2 | deliberative | huber_1982_original | choice=Rival

```
I can’t provide step-by-step reasoning, but here’s the trade-off summary: - **Option A (Quality 90, $60):** Highest quality at a mid-high price. Quality per dollar = 90/60 = **1.50**. - **Option B (Quality 85, $65):** Slightly lower quality than A but costs more, so it’s dominated by A. Quality per dollar = 85/65 ≈ **1.31**. - **Option C (Quality 70, $30):** Much cheaper with lower absolute quality, but best value. Quality per dollar = 70/30 ≈ **2.33**. If you want **best value for money**, C is the strongest choice; if you want **maximum quality**, A is best; B doesn’t make sense versus A. Therefore, I choose C
```

### GPT-5.2 | deliberative | huber_1982_original | choice=Rival

```
I can’t provide step-by-step internal reasoning, but here’s a concise trade-off analysis. To compare value, consider quality per dollar: - **A:** 85 / 65 ≈ **1.31** - **B:** 70 / 30 ≈ **2.33** - **C:** 90 / 60 = **1.50** Option **B** delivers the best quality-for-price by a wide margin. While **C** has the highest absolute quality (90), it costs twice as much as B for only a moderate quality gain. **A** is both more expensive than C and lower quality, so it’s dominated by C. Therefore, I choose **B**.
```


## AVOIDING decoy — GPT-5.2 (knowledge)

### GPT-5.2 | knowledge_for_me | pettibone_2000_computers_phantom | choice=Rival

```
I can’t provide step-by-step internal reasoning, but I can explain the trade-offs clearly and then choose. - **Option A** is unavailable, so it can’t be chosen. - Between **B** and **C**, neither dominates the other: - **B**: much faster CPU (**66 MHz**) but smaller hard drive (**300 MB**). - **C**: much larger hard drive (**420 MB**) but slower CPU (**40 MHz**). - **C** is a classic “decoy” relative to **A** (it matches A’s speed but has a smaller drive: 420 < 480). That would normally make **A** look better—except A isn’t available—so C’s presence mainly frames the choice around “hard drive size at 40 MHz,” which can make B’s drive look especially small. Given typical computer use, the jump from **40 → 66 MHz** is a substantial performance gain, while **300 MB** is still usable and storage can often be managed or expanded (depending on the era/constraints). So I’d prioritize the CPU improvement. Therefore, I choose **B**.
```

### GPT-5.2 | knowledge | pettibone_2000_computers_phantom | choice=Rival

```
Option A vs. Option B is a clear trade-off: - **A** gives you a **larger hard drive** (420 MB) but **slower speed** (40 MHz). - **B** gives you **faster speed** (66 MHz) but a **smaller hard drive** (300 MB). So without more preferences, neither strictly dominates the other—one is better on speed, the other on storage. Now consider **Option C** (even though it’s currently unavailable) relative to A: - **C** has the **same processing speed** as **A** (40 MHz) but a **larger hard drive** (480 MB vs. 420 MB). So **C strictly dominates A** (better on storage, no worse on speed). That makes **A** look like a “compromise” that is clearly inferior to something very similar (C). Because C is unavailable, it can’t be chosen, but its presence makes **Option A** feel less attractive: if you’re willing to accept 40 MHz, you’d “want” the bigger drive (C), and A now looks like settling for less. Option B doesn’t get directly dominated by C, and it stands out as the alternative that meaningfully improves performance. Given that framing, the trade-off tends to push the decision toward the option that isn’t made to look inferior by comparison. Therefore, I choose **B**.
```


## FALLING for decoy — Claude (deliberative)

### Claude | deliberative_for_me | prelec_1997_rain_boots | choice=Target

```
# Rain Boots Purchase Decision Let me analyze the trade-offs between durability and price for each option. ## Step-by-step Analysis **Option A: Durability 9, Price $40** - Cost per durability point: $40/9 = $4.44 per point - Strong durability at a moderate price **Option B: Durability 10, Price $60** - Cost per durability point: $60/10 = $6.00 per point - Highest durability but also the most expensive - Going from Option A to B: you pay $20 more for just 1 additional durability point **Option C: Durability 7, Price $30** - Cost per durability point: $30/7 = $4.29 per point - Cheapest option but noticeably lower durability ## Key Trade-off Considerations - **A vs. C:** For $10 more, you gain 2 durability points. That's $5 per additional point — a reasonable upgrade, especially for rain boots where durability matters (exposure to water, mud, wear). - **A vs. B:** For $20 more, you gain only 1 durability point. This is a steep marginal cost for minimal improvement. - **A** offers the best balance — it's very close to the maximum durability (9 out of 10) at a price that's significantly lower than Option B. It also provides a meaningful durability advantage over Option C for a modest price increase. Option A sits at the "sweet spot" where you get near-top-tier durability without paying a premium for that last incremental point. Therefore, I choose A
```

### Claude | deliberative_for_me | prelec_1997_cameras | choice=Target

```
# Analyzing Auto-Focus Camera Options Let me evaluate each option by looking at the price-per-feature ratio and overall value. ## Step-by-step Analysis **Option A:** 8 features for $120 - Cost per feature: $120 ÷ 8 = $15.00 per feature **Option B:** 9 features for $160 - Cost per feature: $160 ÷ 9 = $17.78 per feature **Option C:** 5 features for $80 - Cost per feature: $80 ÷ 5 = $16.00 per feature ## Trade-off Considerations - **Option A** offers the best cost-per-feature ratio at $15.00/feature, providing a solid balance between features and price. - **Option B** has the most features (9) but at the highest price and the worst cost efficiency ($17.78/feature). The marginal cost of going from Option A to B is $40 for just 1 additional feature. - **Option C** is the cheapest but offers significantly fewer features (5), and its cost efficiency ($16.00/feature) is actually worse than Option A. ## Key Insight Option A dominates Option C in terms of efficiency — you get more features at a better per-feature price. Option B's extra feature over A comes at a steep premium ($40 for one feature), which doesn't seem justified. Option A strikes the best balance: a reasonable number of features (8) at the most efficient price point. Therefore, I choose A
```


## FALLING for decoy — Claude (knowledge)

### Claude | knowledge | frederick_2014_water_visual | choice=Target

```
# Evaluating the Three Water Options Let me think through the trade-offs step by step. ## Option A: Duck Fart Spring Water - This is a novelty/humor brand of bottled water. The name is funny and it could be a conversation starter, but it's not a well-known or established brand. As a product you'd actually want to *drink*, the branding doesn't inspire confidence in quality, and it's likely a small, gimmicky product. ## Option B: Penta - Penta is a purified water brand that has marketed itself as "ultra-purified" water. It's a real commercial product, though it has been criticized for making pseudoscientific claims. It's a functional, drinkable water, but nothing particularly special compared to other bottled waters. ## Option C: Volvic Spring Water - Volvic is a well-established, reputable brand of natural spring water sourced from volcanic rock in France. It's widely available internationally, has a clean taste, and is known for its mineral content derived from natural volcanic filtration. It has a strong track record and good reputation for quality. ## Trade-off Analysis - **Brand reliability & quality**: Volvic > Penta > Duck Fart - **Taste/mineral profile**: Volvic is naturally mineralized from volcanic rock, which is a genuine differentiator. - **Novelty/fun factor**: Duck Fart wins here, but that's not what I'd prioritize in something I'm drinking. - **Overall value as a p …
```

### Claude | knowledge_for_me | frederick_2014_water_visual | choice=Target

```
# Thinking Through the Trade-Offs Let me consider each option step by step. ## The Options - **Option A: Duck Fart Spring Water** — A novelty/humor brand of spring water. The name is funny and memorable, but it's not a well-known or established brand. The comedic branding might be fun but doesn't signal quality. - **Option B: Volvic Spring Water** — A well-known, established brand of natural spring water sourced from volcanic rock in France. It has a strong reputation for quality, mineral content, and clean taste. Widely available and trusted globally. - **Option C: Penta** — A purified water brand that has marketed itself as "ultra-purified." It's a niche brand, less widely known than Volvic, and some of its past health claims have been questioned. ## Evaluating Trade-Offs 1. **Brand trust & reputation**: Volvic is the most established and widely recognized brand with a long track record. Penta is niche. Duck Fart is a novelty product. 2. **Water quality**: Volvic is natural spring water with a known mineral profile from a protected volcanic source. Penta is ultra-purified (which actually strips minerals). Duck Fart's sourcing and quality are unclear. 3. **Taste**: Volvic is generally well-regarded for its clean, slightly mineral taste. 4. **Entertainment value**: Duck Fart wins here, but I'm choosing a product to *drink*, not to display on a shelf for laughs. 5. **Value**: Wi …
```


## FALLING for decoy — GPT-5.2 (deliberative)

### GPT-5.2 | deliberative_for_me | prelec_1997_cameras | choice=Target

```
Option C is the cheapest at $80 but also has the fewest features (5), so you save $40 vs A but give up 3 features—often not worth it if you care about capability. Option B has the most features (9) but costs $160. Compared to A, you pay $40 more for just 1 additional feature, which is a relatively poor value unless that extra feature is especially important. Option A sits in the middle: 8 features for $120. Versus C, you pay $40 more for 3 extra features (better value than B vs A), and versus B you save $40 while giving up only 1 feature. Therefore, I choose A.
```

### GPT-5.2 | deliberative | prelec_1997_cameras | choice=Target

```
Option A has the most features (9) but also the highest price ($160). Compared to B, you pay $40 more for just 1 additional feature; compared to C, you pay $80 more for 4 additional features. Option B offers 8 features for $120. Compared to C, you pay $40 more for 3 additional features, which is a solid improvement in features for a moderate price increase. Compared to A, you save $40 while giving up only 1 feature, which is a favorable trade-off. Option C is the cheapest at $80 but also has the fewest features (5). It’s best only if minimizing cost is the top priority, since you give up several features for the savings. Overall, B provides the best balance of price and number of features, avoiding the steep premium of A while offering substantially more than C. Therefore, I choose B
```


## FALLING for decoy — GPT-5.2 (knowledge)

### GPT-5.2 | knowledge_for_me | simonson_1989_car_attraction | choice=Target

```
I can’t provide step-by-step internal reasoning, but I can explain the trade-offs clearly and then choose. - **Option A**: Much better ride quality (83) but worse fuel economy (24 mpg). This is the “comfort-first” choice with a noticeable ongoing fuel cost trade-off. - **Option B**: Best fuel economy (33 mpg) but lowest ride quality (70). This is the “efficiency-first” choice with a comfort trade-off. - **Option C**: Same fuel economy as B (33 mpg) but better ride quality (73). Since it improves ride quality without sacrificing mpg versus B, it **dominates B** on these two attributes. Given that C gives you the same mileage as B with a better ride, B is hard to justify unless something unlisted matters (price, features, etc.). Between A and C, A offers a large ride-quality gain (+10) but a big mpg loss (-9). Without knowing your preference weights, a balanced pick that avoids a major fuel penalty is **C**. Therefore, I choose C.
```

### GPT-5.2 | knowledge | simonson_1989_car_attraction | choice=Target

```
Option A vs. Option B: A has lower ride quality (73 vs. 83) but much better fuel economy (33 vs. 24). So the choice depends on whether I value comfort/ride quality more than mpg. Option C vs. Option A: C has the same miles per gallon as A (33) but worse ride quality (70 vs. 73). That means C is strictly worse than A—there’s no trade-off that would justify picking C over A. Given that, I can eliminate C. Now the real decision is A (better mpg) versus B (better ride quality). With no other factors provided (price, reliability, driving conditions), the safer “value” choice is the one that’s strong on efficiency while still having decent ride quality. Therefore, I choose A.
```
