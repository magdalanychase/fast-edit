### fast edit
An AI model made for fast code edits

Inspiration from [morphllm](https://morphllm.com/)

Model process:
1. The model gets a code input
2. The input is split into tokens
```typescript
const input = "This is a test";
const parts = input.match(/(\s*\S+)/g);
// parts: [ 'This', ' is', ' a', ' test' ]
const tokensMap = parts.map((part, index) => [`<${index}>`, part])
const tokenizedInput = parts.map(([_, index]) => `<${index}>`)
// [ '<0>', '<1>', '<2>', '<3>' ]
```
3. The tokens are assigned nodes
```typescript

```
4. The nodes are converted to an output
```typescript
// x is the input, y is the output, and W is the weight which transforms the input to the output
y = W * x
```
5. The output is decoded
```typescript
const decodedOutput = tokenized.map(token => {
  const found = tokensMap.find(([tok]) => tok === token);
  return found ? found[1] : '';
});
```

#### Idea:
What if the tokens where consts or functions or extra

Ex:
```
import React, { useState } from 'react';

export default function App() {
  const [count, setCount] = useState(0);

  const increment = () => setCount(count + 1);
  const decrement = () => setCount(count - 1);
  const reset = () => setCount(0);

  return (
    <div style={{ textAlign: 'center', marginTop: '50px' }}>
      <h1>Counter</h1>
      <h2>{count}</h2>
      <button onClick={increment}>Increment</button>
      <button onClick={decrement} style={{ marginLeft: '10px' }}>Decrement</button>
      <button onClick={reset} style={{ marginLeft: '10px' }}>Reset</button>
    </div>
  );
}
```
Change to:
```
<1> React, <8> <13> <9> from 'react'<12>

<2> default <3> App<10><11> <8>
  <4> [<c1>, <sc1>] = <13><10>0<11><12>

  <4> <5> <14> <10><11> => <sc1><10><c1> + 1<11><12>
  <4> <6> <14> <10><11> => <sc1><10><c1> - 1<11><12>
  <4> <7> <14> <10><11> => <sc1><10>0<11><12>

  return <10>
    <div <16>{{ textAlign: 'center', marginTop: '50px' }}>
      <h1>Counter</h1>
      <h2><8><c1><9></h2>
      <button <15><8><5><9>>Increment</button>
      <button <15><8><6><9> <16>{{ marginLeft: '10px' }}>Decrement</button>
      <button <15><8><7><9> <16>{{ marginLeft: '10px' }}>Reset</button>
    </div>
  <11><12>
<9>
```