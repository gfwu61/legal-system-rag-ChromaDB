
# Corporate root CA certificate here.
```text
RB-RootCA-RSA-G01.crt
        │
        │ signiert
        ▼
RB-Proxy-TLS-CA.crt
        │
        │ signiert
        ▼
Zertifikat für github.com

Example:
RB RootCA RSA G01.crt

# This directory is intentionally excluded from Git.
-----------------------------------------------------
Frage:
RB-RootCA-RSA-G01.crt
        │
        │ signiert
        ▼
RB-Proxy-TLS-CA.crt
        │
        │ signiert
        ▼
Zertifikat für github.com
erkläre wie diese Kette funktioniert?  Proxy macht nicht immer TLS? warum braucht man die drei Zertifikate?
```


Antwort:

Ja. Der entscheidende Punkt ist: **Der Proxy macht TLS nicht einfach „immer“ für jede Verbindung.** Bei deinem Bosch-PX-Setup ist relevant, ob **TLS/HTTPS-Inspection** aktiviert und für die Zielverbindung angewendet wird.

Die Zertifikatskette solltest du am besten in zwei getrennte TLS-Verbindungen aufteilen.

## 1. Ohne TLS-Inspection: nur eine TLS-Verbindung

Normalerweise bei:

```text
Python/httpx ───────────────► github.com
             HTTPS / TLS
```

gibt es **eine** TLS-Verbindung.

Die Kette könnte vereinfacht so aussehen:

```text
github.com Zertifikat
        │
        │ signiert von
        ▼
DigiCert Intermediate CA
        │
        │ signiert von
        ▼
DigiCert Root CA
```

Dein Python-System prüft:

> Kann ich dieser Root CA vertrauen?

`certifi` enthält solche öffentlichen Root CAs.

---

# 2. Mit Bosch TLS-Inspection sieht es anders aus

Jetzt kommt der entscheidende Unterschied.

Der Proxy sitzt dazwischen:

```text
Python
   │
   │ TLS-Verbindung 1
   ▼
Bosch PX Proxy
   │
   │ TLS-Verbindung 2
   ▼
github.com
```

Es existieren also **zwei getrennte TLS-Verbindungen**.

### Verbindung 1

```text
Python
   │
   │ HTTPS
   ▼
Bosch PX Proxy
```

### Verbindung 2

```text
Bosch PX Proxy
   │
   │ HTTPS
   ▼
github.com
```

Der Proxy ist dabei TLS-Endpunkt der ersten Verbindung und TLS-Client der zweiten Verbindung.

---

# 3. Was passiert bei `https://github.com`?

Nehmen wir an, dein Python-Code macht:

```python
client.get("https://github.com")
```

### Schritt 1

Python möchte `github.com` erreichen.

Der Request geht an den Proxy:

```text
Python
   │
   │ CONNECT github.com:443
   ▼
PX Proxy
```

Der Proxy baut anschließend selbst eine Verbindung zu GitHub auf.

---

# 4. Die zweite Verbindung

Der Proxy verbindet sich mit GitHub:

```text
PX Proxy ───────── TLS ─────────► github.com
```

GitHub präsentiert sein echtes Zertifikat:

```text
github.com
    │
    ▼
DigiCert ...
    │
    ▼
öffentliche Root CA
```

Der **Proxy** muss dieser öffentlichen CA vertrauen.

Dafür braucht der Proxy seine eigene CA-Konfiguration bzw. den entsprechenden Trust Store.

Das ist eine Sache zwischen:

```text
PX Proxy ↔ GitHub
```

---

# 5. Was sieht dein Python-Programm?

Hier kommt der wichtige Punkt.

Python bekommt bei TLS-Inspection **nicht einfach das originale GitHub-Zertifikat**.

Der Proxy erzeugt bzw. präsentiert für die erste TLS-Verbindung ein Zertifikat, das sinngemäß lautet:

```text
Subject:
    github.com

Issuer:
    RB-Proxy-TLS-CA
```

Also sieht Python:

```text
github.com
     │
     │ issued by
     ▼
RB-Proxy-TLS-CA
```

Python fragt:

> Warum soll ich `RB-Proxy-TLS-CA` vertrauen?

---

# 6. Jetzt kommt `RB-RootCA-RSA-G01`

Die Firmen-PKI hat typischerweise eine Hierarchie:

```text
RB-RootCA-RSA-G01
        │
        │ signiert
        ▼
RB-Proxy-TLS-CA
        │
        │ signiert
        ▼
github.com
```

Das bedeutet:

### 1. Root CA

```text
RB-RootCA-RSA-G01
```

ist die **Vertrauensanker-CA**.

Dein Python-Client vertraut dieser Root CA, weil du sie in dein CA-Bundle aufgenommen hast.

---

### 2. Intermediate CA

```text
RB-Proxy-TLS-CA
```

ist eine untergeordnete CA.

Sie darf Zertifikate für TLS-Inspection ausstellen.

Sie wurde von:

```text
RB-RootCA-RSA-G01
```

signiert.

---

### 3. Server-Zertifikat

Der Proxy kann dann ein Zertifikat für:

```text
github.com
```

präsentieren.

Dieses Zertifikat wird von:

```text
RB-Proxy-TLS-CA
```

signiert.

Damit kann Python die komplette Kette überprüfen:

```text
github.com
    │
    │ signiert von
    ▼
RB-Proxy-TLS-CA
    │
    │ signiert von
    ▼
RB-RootCA-RSA-G01
    │
    │ vertrauenswürdig?
    ▼
JA
```

---

# 7. Warum braucht man überhaupt drei Zertifikate?

Hier muss man etwas präzisieren:

**Dein Python-Client muss nicht notwendigerweise alle drei Zertifikate als lokale Trust Anchors besitzen.**

Der eigentliche Trust Anchor ist normalerweise:

```text
RB-RootCA-RSA-G01
```

Das Intermediate:

```text
RB-Proxy-TLS-CA
```

wird häufig vom Server/Proxy als Teil der Zertifikatskette geliefert.

Das konkrete Verhalten hängt von der Firmen-PKI und der Proxy-Konfiguration ab.

Dein Kombi-Bundle enthält aber vorsichtshalber:

```text
certifi
+
RB-Proxy-TLS-CA
+
RB-RootCA-RSA-G01
```

Das macht dein CA-Bundle robust gegenüber der konkreten Chain-Präsentation.

---

# 8. Wichtig: Proxy macht nicht „immer TLS“

Nein.

Ein Proxy kann verschiedene Dinge tun.

Bei einer normalen HTTP-Verbindung:

```text
Python ── HTTP ──► Proxy ── HTTP ──► Server
```

Bei HTTPS ohne Inspection:

```text
Python
   │
   │ CONNECT
   ▼
Proxy
   │
   │ verschlüsselter TLS-Tunnel
   ▼
GitHub
```

Der Proxy sieht dabei den verschlüsselten TLS-Datenverkehr nicht im Klartext.

Bei HTTPS **mit TLS-Inspection**:

```text
Python
   │
   │ TLS #1
   ▼
Proxy
   │
   │ TLS #2
   ▼
GitHub
```

Jetzt kann der Proxy den TLS-Verkehr entschlüsseln und untersuchen.

---

# 9. Das ist der eigentliche Unterschied

### Ohne Inspection

```text
Python                 Proxy                 GitHub
  │                      │                     │
  │──── CONNECT ────────►│                     │
  │                      │──── TCP/TLS ──────►│
  │◄════════ TLS ═════════════════════════════►│
  │                                             │
  └──────── Ende-zu-Ende-Verschlüsselung ──────┘
```

Der Proxy ist hauptsächlich ein Tunnel.

### Mit Inspection

```text
Python                  Proxy                 GitHub
  │                       │                      │
  │════ TLS #1 ══════════►│                      │
  │                       │════ TLS #2 ═════════►│
  │                       │                      │
  │                       │◄── echtes GitHub ───│
  │◄── Proxy-Zertifikat ──│                      │
  │                       │                      │
```

Der Proxy sitzt als **TLS-Man-in-the-Middle innerhalb der kontrollierten Firmeninfrastruktur**.

Deshalb braucht dein Python-Client die Firmen-CA.

---

# 10. Was passiert, wenn dein `CERT_FILE` nur `certifi` enthält?

Dann kennt Python beispielsweise:

```text
DigiCert
GlobalSign
Let's Encrypt
...
```

aber nicht:

```text
RB-RootCA-RSA-G01
```

Bei der TLS-Inspection sieht Python:

```text
github.com
     │
     ▼
RB-Proxy-TLS-CA
     │
     ▼
RB-RootCA-RSA-G01
```

Python sagt:

```text
RB-RootCA-RSA-G01
        ↓
nicht in meinem Trust Store
        ↓
❌ CERTIFICATE_VERIFY_FAILED
```

Deshalb brauchst du die Firmen-CA.

---

# 11. Was passiert mit deinem Kombi-Bundle?

Deine Datei:

```text
Company_Internet_Kombi.crt
```

enthält:

```text
┌─────────────────────────────────────┐
│ certifi                             │
│                                     │
│ öffentliche CAs                     │
│ DigiCert                            │
│ GlobalSign                          │
│ Let's Encrypt                       │
│ ...                                 │
├─────────────────────────────────────┤
│ RB-Proxy-TLS-CA                     │
├─────────────────────────────────────┤
│ RB-RootCA-RSA-G01                   │
└─────────────────────────────────────┘
```

Damit kann dein Python-Client **beide Welten** validieren:

```text
                  Python
                    │
             Company_Internet_Kombi
                    │
          ┌─────────┴─────────┐
          │                   │
          ▼                   ▼
    öffentliche CAs       Firmen-PKI
          │                   │
          ▼                   ▼
    normale Websites      TLS Inspection
```

Das ist genau der Grund, warum dein Ansatz mit dem Kombi-CA-Bundle sinnvoll ist.

### Der wichtigste Merksatz

> **Bei TLS-Inspection gibt es zwei TLS-Verbindungen: Python ↔ Proxy und Proxy ↔ Internet. Das GitHub-Zertifikat gehört zur zweiten Verbindung; das von der Firmen-PKI ausgestellte `github.com`-Zertifikat gehört zur ersten Verbindung. Deine Firmen-CA ermöglicht Python, dem Zertifikat des Proxy zu vertrauen.**

----------------------

----------------------
``` text
noch eine  Frage über Kerberos und Authentication von Python code:
ich habe die Erfahrung gemacht: mit https://openai.com/
1.  wenn ich HO arbeite , gehe  über VPN ins Firma-Netz verbinden.  
der Python code läuft mit oder ohne PX ohne Fehler.
2. wenn ich in Office arbeite, Python code läuft nur mit PX.
3. in HO oder Office: Verbindung geht immer über Firma-Proxy( das habe ich ausgefunden?
meine Frage:
1. was macht eigentlich PX genau?
2.  Ich kann nicht daran erinnern ob PX automatisch die Authentication für Python code macht. Ich würde sage Ja.  Ich bin nicht sicher,  Ich glaube ich habe  irgendwie sowas gelesen 
```

```text

Ja — deine Beobachtung ist **sehr plausibel**. Wichtig ist aber, drei Dinge auseinanderzuhalten:

1. **VPN**
2. **PX**
3. **Authentifizierung gegenüber dem Firmenproxy**

PX ist nicht einfach „noch ein Proxy“, sondern kann in einer Unternehmensumgebung als **lokaler Authentifizierungs-/Proxy-Agent** fungieren.

## 1. Was macht PX genau?

In deinem Setup läuft PX offenbar lokal auf:
```


```text
127.0.0.1:3128
```

Dein Python-Code kann deshalb beispielsweise sagen:

```text
Python
  │
  │ HTTP/HTTPS
  ▼
127.0.0.1:3128
  │
  ▼
PX
  │
  │ Firmenproxy / Authentifizierung
  ▼
Bosch Internet Proxy
  │
  ▼
Internet
```

PX ist damit eine **lokale Proxy-Schicht** zwischen deiner Anwendung und der eigentlichen Unternehmensinfrastruktur.

---

# 2. Warum funktioniert es im Homeoffice ohne PX?

Das ist sehr interessant.

Du beschreibst:

### Homeoffice + VPN

```text
Laptop
   │
   ▼
VPN
   │
   ▼
Bosch-Netzwerk
   │
   ▼
Firmenproxy
   │
   ▼
Internet
```

Dein Python-Code funktioniert:

```text
Python
   │
   └──► Internet
```

auch ohne lokalen PX.

Das bedeutet sehr wahrscheinlich:

> Im VPN-Zustand sind die notwendigen Routing-, Proxy- und/oder Authentifizierungsmechanismen bereits durch die Unternehmensnetzwerkumgebung verfügbar.

Das muss nicht bedeuten, dass Python „direkt“ ins Internet geht.

---

# 3. Warum funktioniert es im Office nur mit PX?

Das deutet darauf hin, dass das Office-Netzwerk einen anderen Netzwerkpfad verwendet.

Vereinfacht:

```text
Office
Laptop
  │
  ▼
Firmennetz
  │
  ├──► PX ──► Firmenproxy ──► Internet
  │
  └──► direkter Internetzugang
            ❌
```

PX übernimmt dann wahrscheinlich einen Teil der notwendigen Proxy-/Authentifizierungslogik.

Deshalb:

```text
Python
   │
   ├── PX vorhanden → ✓
   │
   └── PX nicht vorhanden → ✗
```

---

# 4. Macht PX die Kerberos-Authentifizierung?

**Sehr wahrscheinlich ist genau das ein wesentlicher Teil der Funktion von PX.**

Aber hier muss ich zwischen dem, was aus deinem beschriebenen Setup sehr wahrscheinlich ist, und dem, was wir sicher wissen, unterscheiden.

Typischer Unternehmensablauf:

```text
Python
   │
   │ HTTP request
   ▼
PX (localhost:3128)
   │
   │ Proxy Authentication
   │
   │ Kerberos / SPNEGO
   ▼
Firmenproxy
```

PX kann dabei die Authentifizierung gegenüber dem Unternehmensproxy übernehmen.

Das ist gerade der große Vorteil eines lokalen Proxy-Agents.

---

# 5. Warum ist das für Python wichtig?

Nehmen wir an, der Unternehmensproxy verlangt:

```text
Proxy-Authentication:
Negotiate
```

Dann kann der Proxy beispielsweise eine Antwort liefern wie:

```text
407 Proxy Authentication Required
Proxy-Authenticate: Negotiate
```

Das bedeutet:

> „Du darfst den Proxy erst benutzen, wenn du dich authentifizierst.“

---

## Ohne PX

Python/httpx müsste selbst mit dieser Authentifizierung umgehen:

```text
Python
  │
  │ HTTP CONNECT
  ▼
Firmenproxy
  │
  │ 407 Proxy Authentication Required
  ▼
Python
  │
  ├── Kerberos/SPNEGO?
  ├── Windows SSPI?
  ├── Credentials?
  └── Token erzeugen?
```

Das ist genau der Bereich, bei dem du früher Probleme mit

```text
requests_kerberos
httpx-negotiate-sspi
PYSPNEGO_NATIVE
```

hattest.

---

# 6. Mit PX sieht es anders aus

Dann:

```text
Python
   │
   │ normaler HTTP request
   ▼
PX :3128
   │
   │ PX übernimmt Proxy-Authentication
   │
   │ Kerberos / SPNEGO / Windows credentials
   ▼
Firmenproxy
   │
   ▼
Internet
```

Dein Python-Code muss dann **nicht unbedingt selbst Kerberos implementieren**.

Das ist vermutlich genau die Erfahrung, die du gemacht hast.

---

# 7. Warum kann PX deinen Windows-Login verwenden?

In einer Windows-Domain-Umgebung existiert typischerweise bereits eine Kerberos-Identität/Ticket-Information für den angemeldeten Benutzer.

Konzeptionell:

```text
Windows Login
     │
     ▼
Active Directory
     │
     ▼
Kerberos Ticket
     │
     ▼
PX
     │
     ▼
Proxy Authentication
```

Dadurch muss PX nicht unbedingt dein Passwort kennen.

Es kann die vorhandenen Windows-/Kerberos-Mechanismen verwenden.

**Das ist ein wichtiger Unterschied:**

PX bekommt nicht notwendigerweise:

```text
username + password
```

sondern kann mit einem vorhandenen Authentifizierungs-Token/Ticket arbeiten.

---

# 8. Das erklärt deine drei Beobachtungen ziemlich gut

Deine Erfahrung:

| Umgebung         |  PX | Python |
| ---------------- | --: | -----: |
| Homeoffice + VPN | aus |      ✅ |
| Homeoffice + VPN |  an |      ✅ |
| Office           |  an |      ✅ |
| Office           | aus |      ❌ |

Das spricht stark dafür, dass die Netzwerkumgebung unterschiedliche Authentifizierungs-/Proxy-Pfade bereitstellt.

Vereinfacht:

```text
                    ┌── Homeoffice + VPN ──► Firmenproxy
                    │
Python ──► Netzwerk ┤
                    │
                    └── Office ──► PX ──► Firmenproxy
```

---

# 9. „Aber wenn beide über den Firmenproxy gehen, warum brauche ich dann PX im Office?“

Das ist die entscheidende Frage.

Weil:

> **Der Firmenproxy ist nicht dasselbe wie PX.**

PX läuft **auf deinem Rechner**:

```text
127.0.0.1:3128
```

Der Unternehmensproxy ist ein **entfernter Firmenproxy**.

Also beispielsweise konzeptionell:

```text
             dein Rechner
                  │
                  ▼
          PX 127.0.0.1:3128
                  │
                  ▼
          Firmenproxy
                  │
                  ▼
               Internet
```

PX ist also ein **lokaler Vermittler**.

---

# 10. Und was macht PX wahrscheinlich noch?

Je nach konkreter PX-Konfiguration kann es beispielsweise:

* Proxy-Authentication übernehmen
* Kerberos/SPNEGO verwenden
* Windows Integrated Authentication verwenden
* Proxy Requests weiterleiten
* Credentials/Tickets aus dem Windows-Kontext verwenden
* unterschiedliche Proxy-Konfigurationen abstrahieren

Deshalb kann deine Python-Anwendung einfach:

```python
proxy="http://127.0.0.1:3128"
```

verwenden.

Die komplexere Firmenauthentifizierung bleibt außerhalb deiner Python-Anwendung.

---

# 11. Warum hattest du dann `requests_kerberos` und `httpx-negotiate-sspi` ausprobiert?

Das passt sehr gut zu diesem Bild.

Wenn Python **direkt** mit dem Firmenproxy kommuniziert:

```text
Python ─────────► Firmenproxy
```

muss Python eventuell selbst:

```text
Kerberos
SPNEGO
Negotiate
SSPI
```

beherrschen.

Wenn du dagegen:

```text
Python ──► PX ──► Firmenproxy
```

verwendest, kann PX diese Aufgabe übernehmen.

Deshalb kann der Python-Code mit PX wesentlich einfacher aussehen.

---

# 12. Aber eine wichtige Einschränkung

Ich würde nicht behaupten:

> „PX macht definitiv immer Kerberos.“

Dafür müsste man deine konkrete PX-Konfiguration bzw. die Kommunikation analysieren.

Was wir aus deinen Beobachtungen **sehr wahrscheinlich** ableiten können:

```text
PX
 │
 ├── lokaler Proxy
 │
 ├── vermittelt deine HTTP/HTTPS-Verbindungen
 │
 └── ist wahrscheinlich in den Firmen-Authentifizierungsmechanismus
     integriert
```

Ob die konkrete Authentifizierung dabei

```text
Kerberos
SPNEGO
NTLM
Windows SSPI
```

oder eine Kombination davon verwendet, müsste man am tatsächlichen Proxy-Handshake prüfen.

---

## 13. Dein aktuelles `client_factory.py` passt genau dazu

Du machst:

```python
if check_if_px_is_running():
    return f"http://{PX_HOST}:{PX_PORT}"
```

also:

```text
PX vorhanden?
    │
    ├── Ja
    │    ↓
    │  http://127.0.0.1:3128
    │
    └── Nein
         ↓
      COMPANY_PROXY_URL
         │
         └── Direct
```

Das ist für deine Umgebung eine **sinnvolle automatische Strategie**.

Und besonders wichtig:

```python
trust_env=False
```

bedeutet:

> „httpx soll nicht selbst irgendwelche `HTTP_PROXY`, `HTTPS_PROXY` oder andere Proxy-Environment-Variablen übernehmen.“

Du gibst den Proxy explizit vor:

```python
httpx.HTTPTransport(
    proxy=proxy_url,
    verify=ssl_context,
)
```

Damit hast du die Netzwerkarchitektur sehr gut unter Kontrolle.

---

### Kurz gesagt

```text
PX ≠ Firmenproxy
```

sondern:

```text
Python
   ↓
PX (lokal)
   ↓
Firmenproxy
   ↓
Internet
```

und **PX kann in einer Unternehmensumgebung die notwendige Proxy-Authentifizierung, einschließlich integrierter Windows-/Kerberos-Mechanismen, vermitteln bzw. übernehmen**.

Deine Beobachtung „Office → nur mit PX“ ist daher sehr gut mit einem solchen Setup vereinbar.



