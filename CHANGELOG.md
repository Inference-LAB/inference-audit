# Changelog

## [0.1.1] - 2026-09-XX

### Fixed
- Clearer error message when `Auditor.audit()` is called from a top-level
  script on Windows without an `if __name__ == "__main__":` guard, instead
  of an unreadable recursive multiprocessing crash.
- Added missing `Programming Language :: Python :: 3.12` classifier
  (CI already tested against 3.12; PyPI metadata was incomplete).

### Documentation
- README Python examples now include the required `__main__` guard, with
  an explanation of why it's needed.