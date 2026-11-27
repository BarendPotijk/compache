import {writeFileSync} from 'node:fs';
import {configure, fitScenario} from '../site/engine.mjs';
import {applyPreset} from '../site/models.mjs';
import {STARTS} from '../site/starts.mjs';

const config = configure(applyPreset(STARTS[0].scenario, 'sonnet'));
writeFileSync(process.argv[2], `export default ${JSON.stringify(fitScenario(config))};\n`);
