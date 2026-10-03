"""Integrity checks for the HPC workflow; no downloaded data or R required."""
import argparse
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/run_ibdmdb_experiment.py'
spec = importlib.util.spec_from_file_location('ibdmdb_runner', SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RunnerIntegrity(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name)/'experiment'
        self.args = argparse.Namespace(output=self.run, r_lib=self.run/'r_lib',
            split_seed=20261003, n_synthetic=2, icfm_steps=4, icfm_seed=20261003,
            icfm_device='cpu', mbgan_iterations=2, mbgan_seed=1,
            mbgan_device='cpu', threads=2)
        self.experiment = runner.Experiment(self.args)
        self.experiment.inputs.mkdir()
        self.experiment.tree.mkdir()
        for split in ('train','test'):
            (self.experiment.inputs/f'sample_{split}.csv').write_text('0.2,0.8\n0.4,0.6\n')
        (self.experiment.inputs/'taxa.txt').write_text('a\nb\n')
        (self.experiment.inputs/'samples.csv').write_text(
            'split,csv_row,sample_id,study_name\n'
            'train,1,A,HMP_2019_ibdmdb\ntrain,2,B,HMP_2019_ibdmdb\n'
            'test,1,C,HMP_2019_ibdmdb\ntest,2,D,HMP_2019_ibdmdb\n')
        for filename in ('childNode.csv','treeTaxa.csv','outTaxa.csv'):
            (self.experiment.tree/filename).write_text('fixture\n')

    def test_overlap_between_train_and_test_is_rejected(self):
        path = self.experiment.inputs/'samples.csv'
        path.write_text(path.read_text().replace('test,1,C,','test,1,A,'))
        with self.assertRaisesRegex(ValueError,'split membership'):
            self.experiment.prepared()

    def test_foreign_study_is_rejected(self):
        path = self.experiment.inputs/'samples.csv'
        path.write_text(path.read_text().replace('test,1,C,HMP_2019_ibdmdb','test,1,C,HMP_2019_t2d'))
        with self.assertRaisesRegex(ValueError,'different study'):
            self.experiment.prepared()

    def test_changed_completed_inputs_are_rejected(self):
        marker = self.run/'complete.json'
        self.experiment.complete(marker, {'seed':1}, self.experiment.real_files())
        (self.experiment.inputs/'sample_train.csv').write_text('0.1,0.9\n0.3,0.7\n')
        with self.assertRaisesRegex(ValueError,'changed or missing'):
            self.experiment.cached(marker, {'seed':1})

    def test_neural_budget_changes_invalidate_completed_model(self):
        fingerprint = self.experiment.model_fingerprint('icfm')
        self.experiment.complete(self.run/'generator_icfm_complete.json',fingerprint,[])
        self.args.icfm_steps = 20000
        with self.assertRaisesRegex(ValueError,'settings changed'):
            self.experiment.cached(self.run/'generator_icfm_complete.json',
                                   self.experiment.model_fingerprint('icfm'))

    def test_invalid_composition_and_taxon_count_are_rejected(self):
        path = self.run/'bad.csv'
        path.write_text('0.2,0.7\n0.4,0.6\n')
        with self.assertRaisesRegex(ValueError,'sum to one'):
            runner.validate_matrix(path)
        path.write_text('0.2,0.8\n0.4,0.6\n')
        with self.assertRaisesRegex(ValueError,'dimensions'):
            runner.validate_matrix(path, columns=77)

    def test_cli_preserves_venv_interpreter_symlink_and_full_budgets(self):
        interpreter = Path(self.temporary.name)/'venv/bin/python'
        interpreter.parent.mkdir(parents=True)
        interpreter.symlink_to('/usr/bin/python3')
        with patch.object(runner, 'Experiment') as factory, patch('sys.argv',
            ['runner','--stage','prepare','--python-generators',str(interpreter)]):
            runner.main()
        args = factory.call_args.args[0]
        self.assertEqual(args.python_generators,interpreter.absolute())
        self.assertEqual(args.icfm_steps,20000)
        self.assertEqual(args.mbgan_iterations,500000)
        self.assertEqual(args.n_synthetic,1000)


if __name__ == '__main__':
    unittest.main()
