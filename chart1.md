flowchart TD
    A[🏥 GLIOMA System] --> B{🔐 Authentication}
    B -->|Register| C[📝 New Account]
    B -->|Login| D[🔑 Existing User]
    C --> E[✅ Dashboard]
    D --> E
    E --> F[🖼️ Upload Brain Scan]
    F --> G[🎯 AI Analysis]
    G --> H[📄 Medical Report]

    classDef system fill:#1e40af,stroke:#ffffff,stroke-width:3px
    classDef process fill:#059669,stroke:#ffffff,stroke-width:2px
    classDef auth fill:#dc2626,stroke:#ffffff,stroke-width:2px

    A system
    C,D,F,G auth
    G --> H[📄 Medical Report]
