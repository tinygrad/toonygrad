Datacenters as (big) brains.
* What will they think about?
* Who controls them? (the market, of course)

How much control do we get?
* The best we can hope for is a free market.
* "Your guy" won't be the winner
* How to we promote the free market? How do we structurally disfavor rent seeking?

If you have a big GPU cluster, what do you do with it?
* Whatever is the most profitable.
* central planning / free market (spoiler: free market wins)

10,000 brains in the world, what do you they think about?

stack:
* silicon (20%)
* TSMC (20%)
* NVIDIA (90%) <-- very high
* Azure (30%)
* OpenAI (10%) <-- VC subsidized (won't stay, will get worse)
4x goes to margins. CHOMP CHOMP CHOMP

-------

The Goal:

Build a large, free, fair, and simple to access market for thought.

tinygrad is a:
* language to express *entire* machine learning jobs concisely
* a schedule, to determine the order of the sub-jobs
* a compiler, to compile these sub-jobs to run on real hardware
* a runtime, to actually run the jobs

llama-3 training job:
* 15,500 steps
* 4M batch size
* 7.7M GPU hours (~$15M)
* only like 10kB to spec the whole $15M job!!
* output is a single hash (plus the value being accessible of course)

imagine submitting that job to a marketplace

------

A big reason the internet sucks today is the "cost to maintain infrastructure"

Bring that convenience to all players, even the smallest.
Remove infrastructure advantage and save the internet.
Don't allow rent seeking on infrastructure this time.

"Our mission is to commoditize the petaflop."

------

AI infra must be taken for granted as free, in the same way Linux is.
History is the story of infrastructure improvement.

------

You can't be everything. "Play my little part in something big"

Create the economic gradient for others to profit off of your work.

tiny franchise:
1. Buy tinyboxes
2. Set them up somewhere
3. Pay franchise fees (small!) and follow guidelines to be part of the tiny network
4. CLOUD=1 will route jobs to you if you are offering the best deal
5. Passive income for you!

We have to build the first test franchise.

No crypto. No token. No "decentralized" like this.

------

Storage. Big hash-value store (key-value store where the key is the hash of the value).

Content addressible Tensors (like, for example, the LLaMA weights)

------

Compete with bigco datacenters.

1. Use whatever GPUs make sense. (language abstracts this. 4090 vs H100 vs 5090 vs B200 vs MI300X vs 7900XTX vs tinychip)
2. Use whatever cooling makes sense. (less uptime is okay, delay or resubmit. tier 0 datacenter)
3. Be whereever. (for training, more latency is fine. places like iceland viable)

This is not a new market. This market exists, we just need to be simpler and cheaper.


It's going to take years....years....YEARSSSSS

------

How do we fail?

* tinygrad never exceeds the speed of PyTorch and never gets adoption
  * be faster than pytorch at inference and training
* Researchers move to JAX instead of tinygrad
  * be a cleaner way to express things
* companies like OpenAI/Anthropic build AGI really fast so this doesn't matter (LOL)

focus on comparative advantage
* we shouldn't make chips. we should make a plan for others to plan chips
* we shouldn't make datacenters. we should make a plan for others

Imagine 27 Chinese chip makers all competing! Imagine 1,000s of franchise datacenters all joining the biggest distributed brain ever, while also having this brain accessible to all size jobs.

The best distribution of AI compute we can hope for is the free market. No backroom deals (but no special win for you and your pet projects or ideology either). This happens if there's real diverse competition.

------

We are in one of three worlds.

1. We're dead no matter what.
2. We're dead or alive based on the decisions we make.
3. We're alive no matter what.

Must act as if in world 2.

Build tech that's inextricable from its narrative.
Tech such that bad guys can only do what you (basically) want with it.
