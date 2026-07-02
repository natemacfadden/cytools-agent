# Verbose Design Log

The repo is small on purpose. I tried a bunch of different things that ended up not working, so I cut them to keep the repo trim. Unfortunately I didn't keep the best record of these experiments (other than my git logs) so I recreated some of the history here

**Original goal**: free, local AI agent that's competitive with frontier models for helping with CYTools computations.

No way I'm beating frontier models more generally. Both in their engineering (I'm just one man...) and in the resources needed to run them. These models seem to have poor knowledge of CYTools, so maybe I can help the user-base by building a *very specialized* harness for driving those computations.

In large part, this project was actually to learn how harnesses work.

# Early Stages (L2)

Much of the early stages were learning the basics like how to drive a model locally (Ollama), how to give these models access to tools, how these models learn to call tools, what an MCP server is etc. Just learning. ALmost immediately I was confronted with the engineering problem of writing a good system prompt and collection of tools.

I took much of this early guidance from what I've heard about frontier models: lean system prompt and minimal tools. Now I know that's a bit biased since these frontier models are general-purpose, but I still think that's the right decision. I justify this with my dumb framing of 'empathy' for the algorithm (please trust me I'm not a crackpot lol). I could give a super long and detailed system prompt, but that's somewhat analogous to asking a new grad student to read a textbook and then asking them to solve research problems. The information dump is hard to parse, contains a lot of noise (details on irrelevant computations to the current task), and difficult to recall.

In favor of this, I aimed to give a lean system prompt that'd be generally valuable to a model. This makes it kinda basic, but it's the information that each agent needs to know (you work on physics, you use CYTools via toolcalls and writing programs, etc.). The detailed information about string theory/CYTools still needs to be given, so that responsibility shifts to tools, error messages, and specialized injection of context.

## Tools

Again following guidance from frontier labs, I aimed to give a thin collection of multi-purpose tools. Instead of "compute the volume of this polytope" I favored "compute general information about the polytope". In this way, by wrapping up N tools into 1, I could reduce the tool surface N-fold. This assumes the model can read its desired data from the output, but that seems to work generally.

No matter how well I engineered the tools, the models still seemed to mess them up. These mistakes were very valuable though because, in nearly all cases, they were human parsable. From dumb things like wanting to read data 'h11' sometimes as `data.h11` and sometimes as `data['h11']` to struggling in switching between derived objects (our pipeline is Polytope->Triangulation->(ToricVariety->; kinda skippable...)CalabiYauManifold->physics). In every case, I still tried to show 'empathy': if the model's call path is clear and unambiguous to me, regardless of whether it is the 'right' path, I generalized the tools to support it.

When the model was doing something truly crazy, this also gave opportunity to provide very directed feedback. In contrast to the system prompt, which is served to all agents regardless of their goal, I have very specific implicit information about what the model is trying to do if it errors-out in any particular tool call. I can infer its goal, understand its error, and provide a very *directed* nudge. This is wher I provide much of the feedback that'd otherwise go in the system prompt, directed nudges when the model struggles. This maintains the connection to how humans best learn: a tutor correcting our mistakes and helping us get beyond them is immeasurably more valuable than a textbook.

## Context

Toolcalling is good, but there is still are a lot of weird technical terms that we use in our group 'divisor volume', 'favorable', 'NTFE', 'automorphisms'. Even if I was provided a gentle/guiding tutor, without knowing the context, I'd be dead in the water. The basic idea was to have an encyclopedia defining certain terms, explaining what they mean, and (in some cases) giving recipes on how to compute them. I did this via regex but now I understand that I was building a rudimentary form of RAG.

## End-Product for Early Stages

At the end of this early stage, I had a harness (toolcalling, context injection) that enabled the model to actually drive CYTools and even make some relevant plots. I tested it on some simple questions that I didn't know the answer to like "For a given N, what is the highest h11 of a polytope such that all of its 2-faces have <= N points" which are relevant to certain downstream applications. These are simple things, but still impressive for a model to be able to do

I'll later work on a 'ladder' of the harness for which this will be rung 'L2'. L1 is reserved for a bare-bones model.

# Orchestration mistake

Partially inspired by the separation of roles in work like https://arxiv.org/html/2605.22763v1, I wanted to try splitting the AI agent into two: a project manager (PM) and an engineer (later renamed to conductor and executor). I had the parallel motivation of wanting to increase trust in the system.

In more detail, I observed models floundering even on some semi-simple multi-stage requests, and they'd hallucinate/lie to give me an answer they thought I wanted. The goal was: if I separate tasks, then one agent (the PM) can just focus on splitting the tasks up into bite-sized chunks while the other agent (the engineer) can approach the smaller problems more easily.

Simultaneous with this split, I introduced a truth ledger. The agents needed to communicate and I was worried (from initial experiments) that the engineer would lie to the PM. If the PM just had to take the engineers word, then the results would be highly untrustworthy. The idea was, since the engineer primarily operates by writing/running code, we could just have this code itself be the message to the PM. Simply capture the written code and the outputs (and optionally some intent/motivation by the engineer) and pass this data to the PM. In this way, one makes it harder (not impossible) for the engineer to lie.

This orchestration made up L3/L4 of the ladder. L3 was the orchestration, L4 was that with a voting scheme. I spent a LOT of time trying to make this work, but (very surprisingly to me), L3&L4 were consistently worse than the simpler L2, despite my models being very simple (e.g., Qwen3:8b). I even had cute engineering ideas in this like using AST to ensure the engineer isn't just faking code outputs (something I observed in initial stages), but ultimately I am abandoning this idea.

As I view it, the major issue is that I added so much beaurocracy that calls became slower and there were more locations to have errors. Even dumb things like "fetch 5 polytopes and tell me how many of them are N-favorable" (conceivable a 2-call answer) faild at the fetching stage. I really tried to force this approach by even adding hand-written deterministic pipelines like "iterate over X, applying Y to it, returning the REDUCTION" which helped reliability in some cases, but this was still spotty (in the models appropriately knowing to call them and giving the right arguments).

# Grading

A frequent issue in my debugging of harness failures was that many of them weren't actually failures in the model/harness, but in the grader. Obviously I can't manually grade all of the responses, so I made a regex method, but that was incredibly brittle. If it was trying to extract a number and there was any other number in the string, it was incredibly difficult to extract the right number. I built a lot of safeguards but those felt very local - just protecting against certain observed issues but not making the larger system more resilient.

I briefly toyed around with using a separately LLM as a judge (throwback to https://arxiv.org/html/2605.22763v1) which actually was much better at extracting the output, but this was a bit unsettling since I don't really trust the LLM. I converged, ultimately, upon enforcing a more-structured/typed output field that is used for grading. I mean this is used for the math HW of millions of students, so why shouldn't it be good here. That did help significantly.
