# Corporate CA Certificates

This directory is intended for **local/company CA certificates** required when running the application inside a corporate network.

## Expected certificates

When running in the corporate environment, the following certificates may be required:

* `RB-RootCA-RSA-G01.crt`
* `RB-Proxy-TLS-CA.crt`

The application automatically detects these certificates and creates a combined CA bundle:

```text
Company_Internet_Kombi.crt
```

If the corporate certificates are not present, the application automatically falls back to the standard CA bundle provided by `certifi`.

## Important: Do not commit certificates

Corporate certificates are **local environment files** and must not be committed to Git or uploaded to GitHub.

The `.gitignore` file excludes certificate files in this directory.

Therefore, after cloning the repository, the `certs/` directory may contain only this `README.md` file.

## Local corporate setup

If you are running the application in a corporate environment and your network requires these certificates, place the required certificate files in this directory:

```text
certs/
├── README.md
├── RB-RootCA-RSA-G01.crt
└── RB-Proxy-TLS-CA.crt
```

The application will detect them automatically.

For non-corporate environments and Streamlit Cloud, no certificates are required. The application uses the standard `certifi` CA bundle instead.
