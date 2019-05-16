workflow "Release to pypi" {
  resolves = [
    "mariamrf/py-package-publish-action@master",
  ]
  on = "release"
}

workflow "Test" {
  on = "push"
  resolves = ["Test with tox"]
}

action "Test with tox" {
  uses = "njzjz/actions/tox-conda@master"
  secrets = ["CODECOV_TOKEN", "COVERALLS_REPO_TOKEN"]
}

action "mariamrf/py-package-publish-action@master" {
  uses = "mariamrf/py-package-publish-action@master"
  secrets = ["TWINE_PASSWORD", "TWINE_USERNAME"]
  env = {
    PYTHON_VERSION = "3.7.3"
  }
}
