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

Concretely this is scattered all through the tool layer. A fetched polytope id will answer `id['id']` or `id.ks_ind` by just handing itself back, so a model that treats an id like a record doesn't hit an opaque crash. Keyword arguments get matched to their real names when the model misremembers them. Ask a result dict for a field that isn't there and the error hands back the fields that *are* ("not a field; available keys: ...") instead of a bare KeyError it has to guess its way out of. The one place I don't bend is correctness: a value that's supposed to be an integer gets an integrality check and errors rather than silently rounding, since a quietly-wrong integer is worse than a failure.

When the model was doing something truly crazy, this also gave opportunity to provide very directed feedback. In contrast to the system prompt, which is served to all agents regardless of their goal, I have very specific implicit information about what the model is trying to do if it errors-out in any particular tool call. I can infer its goal, understand its error, and provide a very *directed* nudge. This is wher I provide much of the feedback that'd otherwise go in the system prompt, directed nudges when the model struggles. This maintains the connection to how humans best learn: a tutor correcting our mistakes and helping us get beyond them is immeasurably more valuable than a textbook.

## Context

Toolcalling is good, but there is still are a lot of weird technical terms that we use in our group 'divisor volume', 'favorable', 'NTFE', 'automorphisms'. Even if I was provided a gentle/guiding tutor, without knowing the context, I'd be dead in the water. The basic idea was to have an encyclopedia defining certain terms, explaining what they mean, and (in some cases) giving recipes on how to compute them. I did this via regex but now I understand that I was building a rudimentary form of RAG.

In practice it's a small glossary: each term gets a definition and, where it helps, a short recipe for computing it in CYTools, and I keyword-match the question to decide which entries to inject. The keyword matching is the crude part, and swapping it for real embedding-based retrieval is the obvious next step I haven't taken.

## End-Product for Early Stages

At the end of this early stage, I had a harness (toolcalling, context injection) that enabled the model to actually drive CYTools and even make some relevant plots. I tested it on some simple questions that I didn't know the answer to like "For a given N, what is the highest h11 of a polytope such that all of its 2-faces have <= N points" which are relevant to certain downstream applications. These are simple things, but still impressive for a model to be able to do

I'll later work on a 'ladder' of the harness for which this will be rung 'L2'. L1 is reserved for a bare-bones model.

# Orchestration mistake

Partially inspired by the separation of roles in work like https://arxiv.org/html/2605.22763v1, I wanted to try splitting the AI agent into two: a project manager (PM) and an engineer (later renamed to conductor and executor). I had the parallel motivation of wanting to increase trust in the system.

In more detail, I observed models floundering even on some semi-simple multi-stage requests, and they'd hallucinate/lie to give me an answer they thought I wanted. The goal was: if I separate tasks, then one agent (the PM) can just focus on splitting the tasks up into bite-sized chunks while the other agent (the engineer) can approach the smaller problems more easily.

Simultaneous with this split, I introduced a truth ledger. The agents needed to communicate and I was worried (from initial experiments) that the engineer would lie to the PM. If the PM just had to take the engineers word, then the results would be highly untrustworthy. The idea was, since the engineer primarily operates by writing/running code, we could just have this code itself be the message to the PM. Simply capture the written code and the outputs (and optionally some intent/motivation by the engineer) and pass this data to the PM. In this way, one makes it harder (not impossible) for the engineer to lie.

This orchestration made up L3/L4 of the ladder. L3 was the orchestration, L4 was that with a voting scheme. I spent a LOT of time trying to make this work, but (very surprisingly to me), L3&L4 were consistently worse than the simpler L2, despite my models being very simple (e.g., Qwen3:8b). (I did check this wasn't just my own bugs: a few of the early L3/L4 losses turned out to be artifacts, like a solver/environment conflict, a dropped final-answer emission, or a plan-gate that misread a real tool result as "nothing produced". I fixed those, which recovered the specific questions but didn't flip the overall result.) I even had cute engineering ideas in this like using AST to ensure the engineer isn't just faking code outputs (something I observed in initial stages), but ultimately I am abandoning this idea.

The AST check is a decent example of the whole trap. It catches the lazy fake, `print(15)` where 15 is just the number the model wanted, because a printed bare constant clearly wasn't computed. But a model that wants 15 badly enough writes `x = 15; print(x)`, or a plausible-looking computation rigged to land on 15, and now the ledger honestly records a real execution that's still a lie. I can force the ledger to reflect what actually ran; I can't force what ran to be the right computation. That's the wall the trust idea hit.

As I view it, the major issue is that I added so much beaurocracy that calls became slower and there were more locations to have errors. Even dumb things like "fetch 5 polytopes and tell me how many of them are N-favorable" (conceivable a 2-call answer) faild at the fetching stage. I really tried to force this approach by even adding hand-written deterministic pipelines like "iterate over X, applying Y to it, returning the REDUCTION" which helped reliability in some cases, but this was still spotty (in the models appropriately knowing to call them and giving the right arguments).

The clearest illustration of the tax: ask for Kahler moduli where the divisor volumes are something impossible (say all negative). Plain L2 calls the solver and gets the whole result back, including a loud "did not converge, don't trust these numbers" flag, and correctly says it can't be done. The orchestrated version maps the computation over the ids and pulls just the one volume field out of the result, so the "did not converge" flag never rides along with it, and it happily reports a garbage number. L2 wins here for a dumb reason: it sees the entire tool output, while the pipeline's neat one-field extraction is exactly what throws the warning away.

# Grading

A frequent issue in my debugging of harness failures was that many of them weren't actually failures in the model/harness, but in the grader. Obviously I can't manually grade all of the responses, so I made a regex method, but that was incredibly brittle. If it was trying to extract a number and there was any other number in the string, it was incredibly difficult to extract the right number. The nastiest version was the answer's own scaffolding leaking a matching digit: a polytope named `h11-3_h21-43` literally contains a 3 and a 43, so a question whose answer was 3 could score correct off the name alone, even when the model never actually computed anything. I built a lot of safeguards but those felt very local - just protecting against certain observed issues but not making the larger system more resilient.

I briefly toyed around with using a separately LLM as a judge (throwback to https://arxiv.org/html/2605.22763v1) which actually was much better at extracting the output, but this was a bit unsettling since I don't really trust the LLM. I converged, ultimately, upon enforcing a more-structured/typed output field that is used for grading. Concretely, each answer ends with a little tagged block, `<final>{"kind": "...", "value": ...}`, and the grader just compares that typed value to the truth (exact for integers, a tolerance band for floats, order-insensitive for lists), with no string-scraping at all. I mean this is used for the math HW of millions of students, so why shouldn't it be good here. That did help significantly.

One thing I made myself do before deleting the old regex grader: re-grade every stored answer with both graders and read every case where they disagreed. All of them were the typed grader being right: one where regex wrongly failed a correct answer, a few where it wrongly passed a wrong one off a coincidental digit. Since the switch only ever fixed mistakes, I felt fine ripping the regex path out.
