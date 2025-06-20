# Contributions guidlines

Contributions are warmly welcomed!
Please, follow the steps below to create a pull request to add new features.


## 1. Install dependecies

You will need to install additional dependencies to compile the documentation, to run the tests and the pre-commits

```bash
pip install .[dev]
```

## 2. Add new code

- Fork the repository and create a new branch.
- On the new branch, add the code for a new feature or bugfix.

## 3. Documentation

- If you are adding a new method or changing/extending and existing one with new features, make sure to document it docstring of the class/method.
- Build the documentation via:

```bash
cd docs
make html
```

- To check the documentation you built, open `docs/build/index.html`.

## 4. Tests
- Create new tests to cover the new code that you added.
- Run the tests

```bash
pytests --cov=tgp --cov-report=term-missing --cov-report=html
```
- Check the code coverage report to see if all the code is covered and correctly executed.
- If you are using VSCode, go on `Testing`, select `pytests` and `/tests`as the target folder.

## 5. Pull request.
- Create a PR on Github.
- Describe your changes in detail.