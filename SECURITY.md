# Security Policy

## Supported Versions

The latest released version of this project receives security fixes.

## Reporting a Vulnerability

Please **do not** open public GitHub issues for security vulnerabilities.

Report privately via one of:

- **GitHub Security Advisories** — Security tab → "Report a vulnerability" (preferred)
- **Email** — open.source@ribose.com

We acknowledge reports within 72 hours and aim to ship a fix within 30 days for critical issues. Coordinated disclosure is supported.

## Disclosure

Public disclosure happens after a fix is released, on a timeline agreed with the reporter.

## Data handling note

This tool is built for license-cleanliness testing: it is designed so
protected-corpus content never enters the repo (environment-path
access only, aggregates-only verdicts, gitignored records dir). If you
find a path by which corpus-derived content could be committed, or a
records/verdict artifact that leaks text, treat it as a security issue
and report it privately per the channels above.
