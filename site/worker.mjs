import {createFitCache, simulate} from './engine.mjs';
import starter from './starter-fit.mjs';

const getFit = createFitCache(starter);
self.onmessage = ({data: {id, config, reportProgress}}) => {
  try {
    const progress = reportProgress ? (candidate, total) => self.postMessage({id, candidate, total}) : undefined;
    const result = simulate(config, settings => getFit(settings, progress));
    self.postMessage({id, result});
  } catch (error) {
    self.postMessage({id, error: error.message});
  }
};
