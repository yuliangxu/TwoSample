"""Exercise submission dependencies and job arguments without a Slurm cluster."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SlurmCommands(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.temp = Path(temporary.name)
        self.log = self.temp/'calls.jsonl'
        self.env = dict(os.environ, SLURM_SUBMIT_DIR=str(ROOT),
            IBDMDB_OUTPUT=str(self.temp/'output with spaces'), MOCK_LOG=str(self.log))

    def fake(self, name, body):
        path = self.temp/name
        path.write_text('#!/usr/bin/env python3\n'+body)
        path.chmod(0o755)
        return path

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_preparation_array_and_analysis_dependencies(self):
        self.fake('sbatch',
            'import json,os,sys\nfrom pathlib import Path\n'
            'p=Path(os.environ["MOCK_LOG"])\n'
            'n=len(p.read_text().splitlines()) if p.exists() else 0\n'
            'with p.open("a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n'
            'print(str(101+n)+";cluster")\n')
        self.env['PATH'] = str(self.temp)+os.pathsep+self.env['PATH']
        subprocess.run(['bash','hpc/submit_ibdmdb.sh'],cwd=ROOT,env=self.env,
                       check=True,capture_output=True,text=True)
        calls = self.calls()
        self.assertEqual(calls[0],['--parsable','hpc/ibdmdb_job.sh','prepare'])
        self.assertIn('--dependency=afterok:101',calls[1])
        self.assertIn('--array=0-3',calls[1])
        self.assertIn('--dependency=afterok:102',calls[2])
        self.assertEqual(calls[2][-1],'analysis')

    def test_array_mapping_and_analysis_stages_preserve_paths_and_budgets(self):
        driver = self.fake('driver',
            'import json,os,sys\n'
            'with open(os.environ["MOCK_LOG"],"a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n')
        self.env['DRIVER_PYTHON'] = str(driver)
        for index in range(4):
            self.env['SLURM_ARRAY_TASK_ID'] = str(index)
            subprocess.run(['bash','hpc/ibdmdb_job.sh','generator'],cwd=ROOT,
                           env=self.env,check=True,capture_output=True)
        subprocess.run(['bash','hpc/ibdmdb_job.sh','analysis'],cwd=ROOT,
                       env=self.env,check=True,capture_output=True)
        calls = self.calls()
        self.assertEqual([c[c.index('--stage')+1] for c in calls],
                         ['d','dt','icfm','mbgan','batts','figures','report'])
        for call in calls:
            self.assertEqual(call[call.index('--output')+1],self.env['IBDMDB_OUTPUT'])
            self.assertEqual(call[call.index('--icfm-steps')+1],'20000')
            self.assertEqual(call[call.index('--mbgan-iterations')+1],'500000')
            self.assertEqual(call[call.index('--n-synthetic')+1],'1000')


if __name__ == '__main__':
    unittest.main()
