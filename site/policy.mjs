// Read a policy that `compache fit` wrote: the decision, what it was fitted on, every candidate it priced and, in newer files, the workload.
import {configurePolicy} from './engine.mjs';
import {presetKey} from './models.mjs';

export const POLICY_SCHEMA = 'compache-workload-planner-v1', AGENT_SCHEMA = 'compache-agent-v1';
export const isPolicy = data => data?.schema_version === POLICY_SCHEMA || data?.schema_version === AGENT_SCHEMA;
const decision = (recipe, pings, long) => configurePolicy({questions: recipe?.input_runs ?? null, answers: recipe?.output_runs ?? null, retrieved: recipe?.retrieval_runs ?? null, keepUnder: recipe?.keep_under_tokens ?? 0, maxPings: pings, long});
export function readPolicy(data) {
  if (data?.schema_version === AGENT_SCHEMA) throw new Error('an agent’s policy decides call by call, and here one decision is made per message: price it with `compache replay`');
  if (!isPolicy(data) || !Array.isArray(data.recipes) || !data.model_spec) throw new Error('this is not a policy written by `compache fit`');
  const recipe = data.recipes.find(r => r.eligible)?.recipe ?? null, solve = data.solve ?? {}, price = data.pricing?.input_price_per_million, turns = solve.turns;
  const per = cost => price && turns && Number.isFinite(cost) ? 1000 * cost / turns * price / 1e6 : null;
  // A policy that asks for a longer cache says what the provider's default is as well.
  const usual = (data.standard_spec ?? data.model_spec).cache_ttl_seconds;
  const policy = decision(recipe, data.planner_options?.max_pings ?? 1, data.model_spec.cache_ttl_seconds > usual), same = (a, b) => ['questions', 'answers', 'retrieved', 'keepUnder', 'maxPings', 'long'].every(key => a[key] === b[key]);
  const candidates = (solve.candidates ?? []).filter(c => c.complete).map(c => ({policy: decision(c.recipe, c.max_pings, (c.cache_ttl_seconds ?? usual) > usual), everything: !c.recipe, coverage: c.coverage, cost: per(c.expected_cost)})).map(c => ({...c, chosen: same(c.policy, policy)}));
  const lives = [usual, Math.max(data.model_spec.cache_ttl_seconds, ...(solve.candidates ?? []).map(c => c.cache_ttl_seconds ?? usual))].map(s => s / 60);
  const notes = [];
  if (recipe?.protect_query_matches) notes.push('It also keeps earlier retrieved text that matches the new question. That is not modeled here.');
  if (recipe?.cited_answer_runs) notes.push('It also keeps the sources that recent answers cited. That is not modeled here.');
  const workload = data.workload && typeof data.workload === 'object' ? data.workload : null;
  return {policy, preset: presetKey(data.pricing?.preset ?? data.model_spec.model_id) ?? null, model: data.model_spec.model_id, threads: data.training_threads?.length ?? data.fitted?.threads ?? null, turns: turns ?? null, coverage: solve.chosen?.coverage ?? null, saving: solve.saving_fraction ?? null,
    cost: per(solve.chosen?.expected_cost), everything: per(solve.keep_everything_cost), candidates, lives, notes, workload};
}
