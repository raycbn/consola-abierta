(() => {
  const grid = document.querySelector('[data-lab-grid]');
  if (!grid) return;
  const cards = Array.from(grid.querySelectorAll('[data-lab-card]'));
  const search = document.querySelector('[data-lab-search]');
  const count = document.querySelector('[data-lab-count]');
  const empty = document.querySelector('[data-lab-empty]');
  const filters = Array.from(document.querySelectorAll('[data-lab-filter]'));
  const state = { topic: 'all', runtime: 'all' };
  const locale = document.documentElement.lang === 'en' ? 'en' : 'es';
  const labels = locale === 'en'
    ? { one: 'lab visible', many: 'labs visible' }
    : { one: 'laboratorio visible', many: 'laboratorios visibles' };
  function update() {
    const term = (search?.value || '').trim().toLocaleLowerCase(locale);
    let visible = 0;
    for (const card of cards) {
      const topicMatch = state.topic === 'all' || card.dataset.topic === state.topic;
      const runtimeMatch = state.runtime === 'all' || card.dataset.runtime === state.runtime;
      const textMatch = !term || (card.dataset.search || card.textContent).toLocaleLowerCase(locale).includes(term);
      const show = topicMatch && runtimeMatch && textMatch;
      card.hidden = !show;
      if (show) visible += 1;
    }
    if (count) count.textContent = `${visible} ${visible === 1 ? labels.one : labels.many}`;
    if (empty) empty.hidden = visible !== 0;
  }
  for (const button of filters) {
    button.addEventListener('click', () => {
      const group = button.dataset.labFilter;
      if (!group || !(group in state)) return;
      state[group] = button.dataset.value || 'all';
      for (const peer of filters.filter(item => item.dataset.labFilter === group)) {
        peer.setAttribute('aria-pressed', String(peer === button));
      }
      update();
    });
  }
  search?.addEventListener('input', update);
  update();
})();
