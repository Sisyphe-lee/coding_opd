from coding_opd.swe_bench_task import verifier_eval_script


def test_verifier_preserves_test_selection_and_patch():
    script = "python -m pip install -e .[test]\n+import pytest\ntox --current-env -epy39 -v -- tests/test_extension.py\n"
    modified = verifier_eval_script(script)
    assert 'pip install --no-build-isolation -e .[test]' in modified
    assert '+import pytest\n' in modified
    assert 'tox --current-env -epy39 -v -- -rA -v --color=no tests/test_extension.py' in modified


def test_official_grader_scores_import_failure_as_unresolved(tmp_path):
    from types import SimpleNamespace
    from swebench.harness.grading import get_eval_report, get_logs_eval

    spec = SimpleNamespace(repo='django/django', version='3.1', instance_id='django-test',
                           FAIL_TO_PASS=['test_fix'], PASS_TO_PASS=['test_existing'])
    log = tmp_path / 'test_output.txt'
    log.write_text(">>>>> Start Test Output\nTraceback (most recent call last):\n"
                   "ImportError: cannot import name 'get_resolver'\n>>>>> End Test Output\n")
    assert get_logs_eval(spec, str(log)) == ({}, True)
    report = get_eval_report(spec, {'instance_id': 'django-test', 'model_patch': 'patch'},
                             str(log), include_tests_status=True)['django-test']
    assert report['patch_successfully_applied']
    assert not report['resolved']
    log.write_text('environment failed before tests started\n')
    assert get_logs_eval(spec, str(log)) == ({}, False)
