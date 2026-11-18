/* Dashboard search adapter. MiniSearch itself is MIT-licensed; see vendor/. */
class DashboardSearch {
  constructor(pages, Engine = MiniSearch) {
    this.pages = new Map(pages.map(page => [page.id, page]));
    this.index = new Engine({
      fields: ['title', 'short_name', 'keywords', 'authors', 'summary', 'metadata', 'body'],
      searchOptions: {
        combineWith: 'AND',
        boost: { title: 6, short_name: 6, keywords: 4, authors: 3, summary: 2, metadata: 1, body: 1 },
      },
    });
    this.index.addAll(pages.map(page => ({ id: page.id, ...this.fields(page) })));
    this.lastQuery = null;
    this.lastResults = [];
  }

  fields(page) {
    return {
      title: page.title || '', short_name: page.short_name || '',
      keywords: [...(page.keywords || []), ...(page.tags || [])].join(' '),
      authors: page.authors || '', summary: [page.hook, page.topic_seed].filter(Boolean).join(' '),
      metadata: [page.doi, page.venue, page.category, ...(page.topics || [])].filter(Boolean).join(' '),
      body: page.body || '',
    };
  }

  query(text) {
    const query = String(text || '').trim();
    if (query === this.lastQuery) return this.lastResults;
    const parts = [...query.matchAll(/"([^"]+)"|(\S+)/g)]
      .map(match => ({ text: (match[1] || match[2]).trim(), quoted: match[1] !== undefined }))
      .filter(part => part.text);
    const phrases = parts.filter(part => part.quoted).map(part => this.normalize(part.text));
    const queries = parts.map((part, position) => ({
      queries: [part.text], combineWith: 'AND',
      prefix: part.quoted ? false : (term, index, terms) =>
        position === parts.length - 1 && index === terms.length - 1 && /^[a-z]{3,}$/.test(term),
      // Short terms and identifiers such as R2, Cas9, and BRCA1 stay exact.
      fuzzy: part.quoted || /^[A-Z\d-]+$/.test(part.text) ? false :
        term => /^[a-z]{6,}$/.test(term) ? 1 : false,
    }));
    let results = queries.length ? this.index.search({ queries, combineWith: 'AND' }) : [];
    if (phrases.length) {
      results = results.filter(result => {
        const fields = Object.values(this.fields(this.pages.get(result.id))).map(value => this.normalize(value));
        return phrases.every(phrase => fields.some(field => field.includes(phrase)));
      });
    }
    this.lastQuery = query;
    this.lastResults = results;
    return results;
  }

  normalize(text) {
    return String(text).toLocaleLowerCase().replace(/\s+/g, ' ').trim();
  }
}

if (typeof module !== 'undefined' && module.exports) module.exports = DashboardSearch;
