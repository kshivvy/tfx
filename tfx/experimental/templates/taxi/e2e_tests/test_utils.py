# Lint as: python2, python3
# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""E2E test utilities for taxi template."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import locale
import os

from absl import logging
from click import testing as click_testing
import tensorflow as tf

from tfx.tools.cli.cli_main import cli_group
from tfx.utils import io_utils


class BaseEndToEndTest(tf.test.TestCase):
  """This test covers step 1~6 of the accompanying document[1] for taxi template.

  [1]https://github.com/tensorflow/tfx/blob/master/docs/tutorials/tfx/template.ipynb
  """

  def setUp(self):
    super(BaseEndToEndTest, self).setUp()

    # Change the encoding for Click since Python 3 is configured to use ASCII as
    # encoding for the environment.
    # TODO(b/150100590) Delete this block after Python >=3.7
    if codecs.lookup(locale.getpreferredencoding()).name == 'ascii':
      os.environ['LANG'] = 'en_US.utf-8'

    self._temp_dir = self.create_tempdir().full_path

    self._pipeline_name = 'TAXI_TEMPLATE_E2E_TEST'
    self._project_dir = os.path.join(self._temp_dir, 'src')
    self._old_cwd = os.getcwd()
    os.mkdir(self._project_dir)
    os.chdir(self._project_dir)

    # Initialize CLI runner.
    self._cli_runner = click_testing.CliRunner()

  def tearDown(self):
    super(BaseEndToEndTest, self).tearDown()
    os.chdir(self._old_cwd)

  def _runCli(self, args):
    logging.info('Running cli: %s', args)
    result = self._cli_runner.invoke(cli_group, args)
    logging.info('%s', result.output)
    if result.exit_code != 0:
      logging.error('Exit code from cli: %d, exception:%s', result.exit_code,
                    result.exception)
      logging.error('Traceback: %s', result.exc_info)

    return result

  def _addAllComponents(self):
    """Change 'pipeline.py' file to put all components into the pipeline."""
    pipeline_definition_file = os.path.join(self._project_dir, 'pipeline',
                                            'pipeline.py')
    with open(pipeline_definition_file) as fp:
      content = fp.read()
    # At the initial state, these are commented out. Uncomment them.
    content = content.replace('# components.append(', 'components.append(')
    io_utils.write_string_file(pipeline_definition_file, content)
    return pipeline_definition_file

  def _copyTemplate(self):
    result = self._runCli([
        'template',
        'copy',
        '--pipeline_name',
        self._pipeline_name,
        '--destination_path',
        self._project_dir,
        '--model',
        'taxi',
    ])
    self.assertEqual(0, result.exit_code)
    self.assertIn('Copying taxi pipeline template', result.output)

