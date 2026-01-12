import React, { ComponentType, useState, ReactNode, useEffect } from "react";

interface Istate {
  Component: ComponentType<any> | null;
}
type TloadRouter = () => Promise<{ default: ComponentType<any> }>;

interface Props {
  comp: React.ElementType; // 👈️ type it as React.ElementType
}
export default function AsyncRouter(loadRouter: TloadRouter) {
  return class Content extends React.Component<any,Istate> {
    constructor(props: any) {
      super(props);
      /* 触发每个路由加载之前钩子函数 */
    }
    state = { Component: null };

    componentDidMount() {
      if (this.state.Component) return;
      loadRouter()
        .then((module) => module.default)
        .then((Component) => this.setState({ Component }));
    }

    render() {
      const { Component } = this.state;
// @ts-ignore
      return Component ? <Component {...this.props} /> : null;
    }
  };
}

interface AsyncRouterProps {
  loadRouter: TloadRouter;
  children?: ReactNode;
}

function withAsyncRouter<P extends object>(WrappedComponent: ComponentType<P>) {
  return function AsyncRouterWrapper(props: P & AsyncRouterProps) {
    const { loadRouter, ...restProps } = props;
    const [Component, setComponent] = useState<ComponentType<P> | null>(null);

    useEffect(() => {
      loadRouter()
        .then((module) => module.default)
        .then((Component) => setComponent(Component));
    }, [loadRouter]);

    if (Component) {
      // @ts-ignore
      return <Component {...restProps} />;
    }
    return null;
  };
}

