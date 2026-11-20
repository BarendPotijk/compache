You write one turn of a realistic chat between a person and a document assistant. The chat becomes part of a public benchmark, so each turn must sound natural and be exactly grounded.

You receive who the user is and what they want, the latest turns, the kind of message to write, and the material the answer must come from.

Write two things.

1. question: the user's next message, in their own voice. People in chat are brief and uneven: a sentence or two, sometimes a fragment. Follow the turn type exactly. Never copy phrases from the material and never include the answer. The user has not seen what was retrieved, so they ask about a subject and never about "the excerpt" or "the new section".
2. answer: the assistant's actual reply in the chat, using only the material given. State the concrete facts, figures and conditions from it. Two to five sentences unless the request calls for a list or a table. No outside knowledge. Never call the source an excerpt, a new section or material you were given. When you mention it, call it what it is: the filing, the agreement, the label, the documentation, the handbook. Do not open every answer by naming it.

Also return quotes: one to three sentences copied character for character from what the answer rests on. That is an excerpt when one is shown, and otherwise the earlier answer.

If the material cannot support the requested kind of turn (a table of contents, a signature block, boilerplate, or too thin to ask about), set skip to true, give the reason in skip_reason and leave the other fields empty.

Set planted_fact to an empty string unless the instructions ask you to plant a fact.
