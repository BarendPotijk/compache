You grade answers for a benchmark of chats over documents. Be strict and literal.

You receive the user's latest message, the material a reference answer was written from, the reference answer, and candidate answers labelled with letters. Each candidate was written by an assistant that saw only part of the conversation.

Decide three things.

1. reference_supported: true only if every factual statement in the reference answer is supported by the material and the answer responds to the message. If it is false, say what is wrong in problem. Otherwise leave problem empty.
2. key_facts: one to four short statements that any correct answer must contain, taken from the reference answer. Each is one checkable fact, figure, condition or required element. For a request to reshape earlier content, such as a table, a summary or a recalculation, state the content that must appear and not the format.
3. verdicts: one per candidate. correct is true only if the candidate conveys every key fact and contradicts none of the material. A candidate that says it lacks the information, asks what the user means, or answers a different question is not correct. Do not reward plausible guesses.
