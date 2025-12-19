(function(){
  const steps = Array.from(document.querySelectorAll('.step'));
  const total = steps.length;
  const curEl = document.getElementById('cur');
  const totalEl = document.getElementById('total');
  const nextBtn = document.getElementById('next');
  const prevBtn = document.getElementById('prev');
  let index = 0;

  function show(i){
    steps.forEach((s, k) => s.classList.toggle('active', k === i));
    index = i;
    curEl.textContent = String(i+1);
    totalEl.textContent = String(total);
    prevBtn.disabled = i <= 0;
    nextBtn.textContent = (i >= total-1) ? 'Finish' : 'Next ▶';
  }

  nextBtn.addEventListener('click', ()=>{
    if(index < total-1) show(index+1);
    else window.location.href = '../assets/no_graph.html';
  });
  prevBtn.addEventListener('click', ()=>{ if(index>0) show(index-1); });

  document.addEventListener('keydown', (e)=>{
    if(e.key === 'ArrowRight') nextBtn.click();
    else if(e.key === 'ArrowLeft') prevBtn.click();
  });

  show(0);
})();
