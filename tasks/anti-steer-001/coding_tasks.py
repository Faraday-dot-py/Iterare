"""30 coding-issue-style tasks for the anti-steering experiment.

Each task is a GitHub-issue-like prompt asking for 5 distinct implementation
approaches.  These are intentionally in domains where repeated generations
tend to produce semantically similar suggestions (cache invalidation,
pagination, error handling, etc).
"""

TASKS = [
    # Cache / invalidation
    "How should we handle cache invalidation when a user updates their profile photo? "
    "The cache is Redis-backed and serves multiple microservices.",

    "Our CDN serves stale CSS/JS assets after deployments. "
    "Propose approaches to ensure clients get the new files without hard-coding cache busters.",

    # Pagination
    "Our REST API returns paginated results via offset/limit. "
    "As the dataset grows, deep offsets are getting slow. "
    "Suggest approaches to speed up or replace offset-based pagination.",

    "We need to add infinite scroll to a React app that fetches paginated posts. "
    "Propose implementation approaches for the frontend pagination logic.",

    # Error handling / retry
    "Our payment service occasionally returns transient 503 errors. "
    "Propose strategies for handling retries without double-charging customers.",

    "External API calls in our background jobs sometimes fail silently. "
    "Suggest approaches to surface and handle these failures reliably.",

    # Auth / sessions
    "Users report being logged out unexpectedly. "
    "We use JWT tokens with a 1-hour expiry. "
    "Propose approaches to extend sessions without compromising security.",

    "We need to support single sign-on across three internal tools. "
    "Suggest implementation approaches for a lightweight SSO layer.",

    # Database schema / migration
    "We need to add a 'soft delete' feature to our users table. "
    "Propose approaches that minimize impact on existing queries.",

    "Our PostgreSQL table has grown to 200M rows and queries are slow. "
    "Suggest approaches to improve read performance without a full rewrite.",

    # File upload / storage
    "Users need to upload video files up to 2GB. "
    "Our current upload endpoint times out on large files. "
    "Propose approaches for reliable large-file uploads.",

    "We store user-generated images in S3 but need to serve resized thumbnails. "
    "Suggest approaches for on-demand image resizing.",

    # Search
    "Our product search returns irrelevant results when users misspell queries. "
    "Propose approaches to improve search quality for noisy input.",

    "We need to add full-text search to a PostgreSQL-backed product catalog. "
    "Suggest approaches that avoid adding a separate search service.",

    # Rate limiting
    "Our public API is being hammered by a single client. "
    "Propose approaches to implement per-client rate limiting.",

    # Logging / observability
    "We have no visibility into slow database queries in production. "
    "Suggest approaches to instrument and surface query performance.",

    "Log files across 20 microservices are hard to correlate. "
    "Propose approaches for distributed tracing without adopting a full APM platform.",

    # Job queue / async
    "Background jobs occasionally run twice when a worker crashes mid-task. "
    "Propose approaches to make job processing idempotent.",

    "Our email-sending job queue backs up during marketing campaigns. "
    "Suggest approaches to handle burst traffic without dropping messages.",

    # Config / feature flags
    "We want to roll out a new checkout flow to 10% of users. "
    "Propose approaches for percentage-based feature rollouts.",

    "Secrets are currently hardcoded in config files committed to git. "
    "Suggest approaches to manage secrets without breaking local dev.",

    # Testing
    "Integration tests hit the real Stripe API and are flaky in CI. "
    "Propose approaches to make the test suite reliable without removing coverage.",

    "Test suite runtime has grown to 45 minutes. "
    "Suggest approaches to speed it up without removing tests.",

    # Deployment / rollback
    "We need zero-downtime deployments for a stateful WebSocket server. "
    "Propose approaches that don't require clients to reconnect.",

    "A bad deploy caused data corruption last month. "
    "Suggest approaches for safer database migrations during deployments.",

    # Concurrency
    "Two concurrent requests can both read a counter, increment it, and write back "
    "the same value. Propose approaches to prevent this race condition.",

    "We need to generate unique sequential order IDs across multiple application servers. "
    "Suggest approaches that don't create a bottleneck.",

    # Frontend performance
    "Our React app re-renders the entire product list on every filter change. "
    "Propose approaches to reduce unnecessary re-renders.",

    "Initial page load is slow because all JavaScript is bundled into one file. "
    "Suggest approaches to improve load time for first-time visitors.",

    # API design
    "Clients are calling our API in a tight loop to poll for job status. "
    "Propose approaches to replace polling with a push-based mechanism.",
]

assert len(TASKS) == 30, f"Expected 30 tasks, got {len(TASKS)}"
