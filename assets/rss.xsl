<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0"
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
  <xsl:output method="html" encoding="UTF-8" indent="yes"/>
  <xsl:template match="/">
    <html lang="{/rss/channel/language}">
      <head>
        <meta charset="utf-8"/>
        <meta name="viewport" content="width=device-width, initial-scale=1"/>
        <title><xsl:value-of select="/rss/channel/title"/> | RSS</title>
        <style>
          :root{color-scheme:dark;--bg:#07111f;--panel:#101e30;--line:#263c55;--text:#e8f0fa;--muted:#9db0c7;--accent:#62d5ff;--accent2:#7c9cff}
          *{box-sizing:border-box}
          body{margin:0;background:radial-gradient(ellipse at 15% 0%,#12304a 0,transparent 42%),var(--bg);color:var(--text);font:16px/1.65 system-ui,-apple-system,"Segoe UI",sans-serif}
          a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
          .shell{width:min(900px,calc(100% - 36px));margin:0 auto;padding:34px 0 56px}
          .top{display:flex;align-items:center;justify-content:space-between;gap:16px;padding-bottom:24px;border-bottom:1px solid var(--line)}
          .brand{display:flex;align-items:center;gap:12px;font-weight:750;letter-spacing:.02em}
          .mark{display:grid;place-items:center;width:42px;height:42px;border:1px solid #3b82a6;border-radius:12px;color:var(--accent);background:#0c263b;font-size:13px}
          .eyebrow{margin:38px 0 8px;color:var(--accent);font-size:12px;font-weight:800;letter-spacing:.16em;text-transform:uppercase}
          h1{margin:0;font-size:clamp(2rem,6vw,3.4rem);line-height:1.08;letter-spacing:-.04em}
          .intro{max-width:680px;margin:14px 0 0;color:var(--muted);font-size:1.08rem}
          .hint{margin:26px 0 22px;padding:14px 17px;border:1px solid var(--line);border-radius:12px;background:#0b1929;color:var(--muted);font-size:.92rem}
          .items{display:grid;gap:16px}
          article{padding:22px 24px;border:1px solid var(--line);border-radius:16px;background:linear-gradient(145deg,rgba(17,34,54,.96),rgba(10,22,37,.96));box-shadow:0 14px 38px rgba(0,0,0,.16)}
          .meta{display:flex;flex-wrap:wrap;gap:10px 16px;align-items:center;color:var(--muted);font-size:.78rem}
          .category{color:var(--accent);font-weight:750;letter-spacing:.08em;text-transform:uppercase}
          h2{margin:10px 0 8px;font-size:clamp(1.2rem,3.4vw,1.55rem);line-height:1.25}
          article p{margin:0;color:#bdcadd}
          .more{display:inline-block;margin-top:16px;font-size:.9rem;font-weight:700}
          footer{display:flex;justify-content:space-between;flex-wrap:wrap;gap:12px;margin-top:30px;padding-top:18px;border-top:1px solid var(--line);color:var(--muted);font-size:.85rem}
          @media(max-width:560px){.shell{width:min(100% - 24px,900px);padding-top:22px}.top{align-items:flex-start}.intro{font-size:1rem}article{padding:18px}.hint{font-size:.86rem}}
        </style>
      </head>
      <body>
        <main class="shell">
          <div class="top">
            <div class="brand"><span class="mark">CA</span><span>Consola Abierta</span></div>
            <a href="{/rss/channel/link}"><xsl:choose><xsl:when test="/rss/channel/language='en'">Visit publication ↗</xsl:when><xsl:otherwise>Visitar la publicación ↗</xsl:otherwise></xsl:choose></a>
          </div>
          <header>
            <p class="eyebrow">RSS · <xsl:value-of select="/rss/channel/language"/></p>
            <h1><xsl:value-of select="/rss/channel/title"/></h1>
            <p class="intro"><xsl:value-of select="/rss/channel/description"/></p>
          </header>
          <p class="hint"><xsl:choose><xsl:when test="/rss/channel/language='en'">Human-readable feed view. Copy this page address into your RSS reader to follow new publications.</xsl:when><xsl:otherwise>Vista legible del feed. Puedes copiar la dirección de esta página y añadirla a tu lector RSS para recibir las nuevas publicaciones.</xsl:otherwise></xsl:choose></p>
          <section class="items">
            <xsl:for-each select="/rss/channel/item">
              <article>
                <div class="meta">
                  <span class="category"><xsl:value-of select="category"/></span>
                  <time><xsl:value-of select="pubDate"/></time>
                </div>
                <h2><a href="{link}"><xsl:value-of select="title"/></a></h2>
                <p><xsl:value-of select="description"/></p>
                <a class="more" href="{link}"><xsl:choose><xsl:when test="/rss/channel/language='en'">Read article ↗</xsl:when><xsl:otherwise>Leer artículo ↗</xsl:otherwise></xsl:choose></a>
              </article>
            </xsl:for-each>
          </section>
          <footer>
            <span><xsl:choose><xsl:when test="/rss/channel/language='en'">Consola Abierta · Engineering &amp; technology</xsl:when><xsl:otherwise>Consola Abierta · Ingeniería y tecnología</xsl:otherwise></xsl:choose></span>
            <span><xsl:value-of select="/rss/channel/lastBuildDate"/></span>
          </footer>
        </main>
      </body>
    </html>
  </xsl:template>
</xsl:stylesheet>
