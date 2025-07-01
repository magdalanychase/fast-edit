const log = (text: string | any) => {
    console.log(text)
}

const text = 'The cat sat'

const words = text.split(' ')

const queries = words
const keys = words
const values = words

const vectors = {
    'The': [1],
    ' cat': [1],
    ' sat': [1],
    ' on': [1],
    ' the': [1],
    ' porch': [1],
    '.': [0]
}

let qs: (string | number)[][] = []
for (const query of queries) {
    let q = 0
    if (words.join(' ').includes(query)) {
        q = 1
    }
    qs.push([query, q])
}

let ks: (string | number)[][] = []
for (const key in vectors) {
    const k = vectors[key][0]
    ks.push([key, k])
}

log('Queries:'+ qs)
log('Keys:' + ks)

let products: (string | number)[][] = []
for (const [query, q] of qs) {
    for (const [key, k] of ks) {
        products.push([`${query} * ${key}`, Number(q) * Number(k)])
    }
}
log('Products:' + products)

