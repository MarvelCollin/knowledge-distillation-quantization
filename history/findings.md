# Key Findings

## Does the student really learn? Yes

The distilled student solves 17 brand new problems the untrained model could not, keeps 19, loses 3 (net +14). Concrete evidence of new skill:

- missing_function failures dropped from 865 to 85 (the untrained model wrote incomplete code that referenced undefined functions; after distillation it writes complete functions).
- runtime errors dropped from 2842 to 2164.
- pass@1 doubled.

So learning clearly happened and it is large.

## Why does one teacher equal two teachers equal about 38?

The limit is the student's capacity, not the teachers.

Think of the 1.5B parameters as a cup and teacher knowledge as water. One teacher already fills the cup to the brim (about 38 solves). Adding a second teacher pours more water into a full cup, it cannot hold more. Proof the cup is full after one teacher: the on policy test showed the teacher agrees with the student's own tokens 96.6% of the time, so almost nothing is left to transfer.

## Where did the 42 truncated problems go?

In the two teacher model the 42 truncated problems became: 29 wrong_answer, 4 now solved, 4 runtime_error, 4 timeout, 1 still truncated. Truncation was mostly happening on problems the student could not solve anyway (hard ones). Fixing the rambling makes the failures clean instead of a 40000 character repetition loop, and quietly recovered 4 solves.

## Student limited or teacher limited?

Neither simply. The teacher is far better (126 vs 36) so it has plenty to teach. But token level distillation is the wrong channel: the student and teacher agree token by token on the student's path (96.6%), yet the student picks worse overall approaches. Token KD fixes local token choices, not the strategic wrong turn. The remaining gap is mostly student capacity: on hard problems the student trained on 283 correct teacher solutions and still solves only 5 of 79, and DeepSeek's own 1.5B distilled with 800000 traces lands at the same level we hit with about 1600.

## Consistency (the lucky solves)

About half the solves are lucky (pass 1 of 5 samples). Temperature sweep:

| Temp | Solved | pass@1 | pass@5 | consistency (p1/p5) |
|---|---|---|---|---|
| 0.6 | 35 | 7.7% | 15.4% | 0.50 |
| 0.4 | 27 | 5.7% | 11.8% | 0.48 |
| 0.2 | 28 | 7.1% | 12.3% | 0.58 |

Lower temperature raises the consistency ratio but loses more absolute solves than it gains. Temperature 0.6 wins on every metric that matters. Consistency is a capacity property (student 0.50 vs teacher 0.70), it scales with model size not decoding.

## What would actually move the number

1. More pure KD on this student: no, proven exhausted.
2. A different method (RL with verifiable rewards, rejection sampling): could add a few points by attacking the strategic axis, but out of the KD only title scope.
3. A bigger student (3B): the real lever, more capacity to hold the strategy the teacher offers. Same teacher, same recipe, cache reusable. Measure the untrained 3B baseline first, because a stronger base can shrink the original to distilled gap even as absolute rises.
