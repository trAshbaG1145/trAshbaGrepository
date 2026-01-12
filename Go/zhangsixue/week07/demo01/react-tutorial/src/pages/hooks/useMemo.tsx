import { useCallback, useMemo, useState } from "react";

const testList = [
  {
    id: 1,
    name: "zhangsan",
  },
  {
    id: 2,
    name: "lisi",
  },
  {
    id: 3,
    name: "wangwu",
  },
  {
    id: 4,
    name: "zhaoliu",
  },
];
export default function HomePage() {
  const [count, setCount] = useState(0);

  console.time("handleClick");
  const t1 = Date.now()
  const handleClick = useCallback((event) => {
    console.log(event.target);
    console.dir(event.target);
    console.log(event.target.dataset.id);
    setCount((v) => v + 1);
  }, []);
  const t2 = Date.now()
  console.timeEnd("handleClick");
  console.log('t2-t1',t2-t1)

  const handleClick2 = useMemo(
    () => (event) => {
      console.log(event.target);
      console.dir(event.target);
      console.log(event.target.dataset.id);
    },

    []
  );

  console.time("handleClick3");
  const handleClick3 = (event) => {
    console.log(event.target);
    console.dir(event.target);
    console.log(event.target.dataset.id);
    setCount((v) => v + 1);
  };
  console.timeEnd("handleClick3");

  return (
    <div>
      <ul>
        {testList.map((v) => {
          return (
            <li key={v.id} data-id={v.id} onClick={handleClick}>
              {v.name}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
